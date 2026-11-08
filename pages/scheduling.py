#JIE42903 – Evolutionary Computing 
#Assignment | TV Scheduling using Genetic Algorithm 

import random
from itertools import permutations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TV Scheduling | Genetic Algorithm",
    layout="wide"
)

# Program ratings for each time slot
program_ratings_dict = {
    "news": [0.1, 0.7, 0.8, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.7, 0.8, 0.4, 0.3, 0.2, 0.1, 0.2],
    "live_soccer": [0.0, 0.0, 0.0, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.3, 0.4, 0.5, 0.4, 0.8, 0.8, 0.9, 0.7, 0.3],
    "movie_a": [0.1, 0.1, 0.2, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.7, 0.8, 0.7, 0.4],
    "movie_b": [0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.7, 0.4, 0.3, 0.7, 0.5],
    "reality_show": [0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.3, 0.4, 0.7, 0.7, 0.3, 0.2, 0.1, 0.2, 0.3],
    "tv_series_a": [0.2, 0.3, 0.2, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.7, 0.8, 0.4, 0.7, 0.4, 0.3],
    "tv_series_b": [0.1, 0.2, 0.3, 0.3, 0.2, 0.3, 0.3, 0.1, 0.4, 0.3, 0.4, 0.3, 0.5, 0.3, 0.7, 0.8, 0.4, 0.3],
    "music_program": [0.6, 0.7, 0.3, 0.2, 0.2, 0.1, 0.2, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.2, 0.3, 0.5, 0.3],
    "documentary": [0.3, 0.3, 0.4, 0.3, 0.6, 0.6, 0.3, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.2],
    "Boxing": [0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.9],
}

# Genetic Algorithm settings
GEN = 100
POP = 50
EL_S = 2

ratings = program_ratings_dict
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))


def fitness_function(schedule):
    total_rating = 0

    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]

    return total_rating


def initialize_pop(programs, time_slots):
    return permutations(programs)


def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)

        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = list(schedule)

    return best_schedule


def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)

    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]

    return child1, child2


def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)

    schedule[mutation_point] = new_program

    return schedule


# Generate the schedule using the selected parameters
def genetic_algorithm(
    initial_schedule,
    generations=GEN,
    population_size=POP,
    crossover_rate=0.8,
    mutation_rate=0.2,
    elitism_size=EL_S
):

    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        population.sort(
            key=lambda schedule: fitness_function(schedule),
            reverse=True
        )

        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)

            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]


@st.cache_data(show_spinner="Preparing initial schedule...")
def find_initial_schedule():
    return finding_best_schedule(
        initialize_pop(all_programs, all_time_slots)
    )


def run_trial(initial_schedule, crossover_rate, mutation_rate):
    remaining_slots = len(all_time_slots) - len(initial_schedule)

    genetic_schedule = genetic_algorithm(
        initial_schedule.copy(),
        generations=GEN,
        population_size=POP,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=EL_S
    )

    final_schedule = (
        initial_schedule
        + genetic_schedule[:remaining_slots]
    )

    total_fitness = fitness_function(final_schedule)

    schedule_table = pd.DataFrame({
        "Time Slot": [
            f"{hour:02d}:00"
            for hour in all_time_slots
        ],
        "Scheduled Program": final_schedule
    })

    return round(total_fitness, 2), schedule_table


# Page layout
st.title("TV Scheduling Algorithm")
st.caption("Genetic Algorithm")
st.divider()

st.subheader("1. Algorithm Parameters")

col1, col2 = st.columns(2)

with col1:
    CO_R = st.number_input(
        "Crossover Rate (CO_R)",
        min_value=0.0,
        max_value=0.95,
        value=0.80,
        step=0.01,
        format="%.2f"
    )

with col2:
    MUT_R = st.number_input(
        "Mutation Rate (MUT_R)",
        min_value=0.01,
        max_value=0.20,
        value=0.20,
        step=0.01,
        format="%.2f"
    )

st.caption("Generations: 100 | Population: 50 | Elitism: 2")

if "scheduling_trials" not in st.session_state:
    st.session_state.scheduling_trials = []


if st.button("Run Genetic Algorithm", type="primary"):

    with st.spinner("Generating TV schedule..."):
        initial_schedule = find_initial_schedule()

        score, schedule_table = run_trial(
            initial_schedule,
            CO_R,
            MUT_R
        )

    st.session_state.scheduling_trials.append({
        "Trial": len(st.session_state.scheduling_trials) + 1,
        "CO_R": CO_R,
        "MUT_R": MUT_R,
        "Total Fitness": score,
        "Schedule": schedule_table
    })


# Display scheduling results
if st.session_state.scheduling_trials:

    latest = st.session_state.scheduling_trials[-1]

    st.divider()
    st.subheader("2. Final TV Schedule")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Fitness",
        f"{latest['Total Fitness']:.2f}"
    )

    col2.metric(
        "Time Slots",
        len(latest["Schedule"])
    )

    st.dataframe(
        latest["Schedule"],
        hide_index=True,
        use_container_width=True,
        height=680
    )

    st.divider()
    st.subheader("3. Experiment History")

    results_table = pd.DataFrame([
        {
            "Trial": item["Trial"],
            "CO_R": item["CO_R"],
            "MUT_R": item["MUT_R"],
            "Total Fitness": item["Total Fitness"]
        }
        for item in st.session_state.scheduling_trials
    ])

    st.dataframe(
        results_table,
        hide_index=True,
        use_container_width=True
    )

    for item in st.session_state.scheduling_trials:

        with st.expander(
            f"Trial {item['Trial']} | "
            f"CO_R: {item['CO_R']:.2f} | "
            f"MUT_R: {item['MUT_R']:.2f} | "
            f"Fitness: {item['Total Fitness']:.2f}"
        ):
            st.dataframe(
                item["Schedule"],
                hide_index=True,
                use_container_width=True,
                height=680
            )

    if st.button("Clear Trial History"):
        st.session_state.scheduling_trials = []
        st.rerun()

else:
    st.info(
        "Set the parameters and click Run Genetic Algorithm "
        "to generate a TV schedule."
    )
