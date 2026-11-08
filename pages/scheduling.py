"""TV program scheduling with a genetic algorithm."""

import csv
import io
import itertools
import random

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TV Scheduling | Genetic Algorithm",
    layout="wide"
)

# Default program ratings for each hourly time slot
DEFAULT_RATINGS = {
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

GEN = 100
POP = 50
EL_S = 2
ALL_TIME_SLOTS = list(range(6, 24))


def read_csv_to_dict(uploaded_file):
    """Read program ratings from a CSV file."""

    text = uploaded_file.getvalue().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    next(reader, None)

    program_ratings = {}

    for line_no, row in enumerate(reader, start=2):
        if not row or not any(cell.strip() for cell in row):
            continue

        if len(row) != 19:
            raise ValueError(
                f"Row {line_no} must contain 1 program name and 18 ratings."
            )

        program = row[0].strip()

        if not program or program in program_ratings:
            raise ValueError(
                f"Invalid or duplicate program name at row {line_no}."
            )

        try:
            values = [float(value) for value in row[1:]]
        except ValueError:
            raise ValueError(
                f"Invalid rating value at row {line_no}."
            )

        if not all(0 <= value <= 1 for value in values):
            raise ValueError(
                f"Ratings must be between 0 and 1 at row {line_no}."
            )

        program_ratings[program] = values

    if len(program_ratings) != 10:
        raise ValueError("The CSV must contain exactly 10 programs.")

    return program_ratings


# Program ratings used in fitness calculations
ratings = DEFAULT_RATINGS
all_programs = list(ratings.keys())


def fitness_function(schedule):
    total_rating = 0

    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]

    return total_rating


def initialize_pop(programs, time_slots):
    # Generate possible program combinations
    return itertools.permutations(programs)


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


def evaluate_fitness(schedule):
    return fitness_function(schedule)


# Genetic Algorithm
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

        # Keep the highest fitness schedules
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


@st.cache_data(
    show_spinner="Finding the best initial schedule..."
)
def find_initial_schedule(ratings_data):
    # Reuse the result when the rating data is unchanged
    global ratings

    ratings = ratings_data

    return finding_best_schedule(
        initialize_pop(
            list(ratings_data.keys()),
            ALL_TIME_SLOTS
        )
    )


def run_trial(initial_best_schedule, crossover_rate, mutation_rate):
    remaining = len(ALL_TIME_SLOTS) - len(initial_best_schedule)

    genetic_schedule = genetic_algorithm(
        initial_best_schedule.copy(),
        generations=GEN,
        population_size=POP,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=EL_S
    )

    final_schedule = (
        initial_best_schedule
        + genetic_schedule[:remaining]
    )

    score = fitness_function(final_schedule)

    schedule_df = pd.DataFrame({
        "Time Slot": [
            f"{hour:02d}:00"
            for hour in ALL_TIME_SLOTS
        ],
        "Scheduled Program": final_schedule
    })

    return round(score, 2), schedule_df


# Main page
st.title("TV Scheduling Algorithm")
st.caption("Genetic Algorithm")
st.divider()


# Dataset
st.subheader("1. Program Ratings")

upload = st.file_uploader(
    "Upload Program Ratings CSV",
    type=["csv"],
    help="The file must contain 10 programs and 18 hourly ratings."
)

try:
    current_ratings = (
        read_csv_to_dict(upload)
        if upload else DEFAULT_RATINGS
    )
except (ValueError, UnicodeDecodeError) as exc:
    st.error(f"Invalid CSV: {exc}")
    st.stop()

st.caption(
    "Using uploaded CSV"
    if upload else "Using default program ratings"
)


# Parameters
st.subheader("2. Algorithm Parameters")

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

st.caption(
    "Generations: 100 | Population: 50 | Elitism: 2"
)


# Reset history when the dataset changes
dataset_key = tuple(
    (key, tuple(values))
    for key, values in current_ratings.items()
)

if st.session_state.get("scheduling_dataset_key") != dataset_key:
    st.session_state.scheduling_dataset_key = dataset_key
    st.session_state.scheduling_trials = []


# Run scheduling
if st.button("Run Genetic Algorithm", type="primary"):

    ratings = current_ratings
    all_programs = list(ratings.keys())

    with st.spinner("Generating TV schedule..."):
        initial = find_initial_schedule(current_ratings)

        score, schedule_df = run_trial(
            initial,
            CO_R,
            MUT_R
        )

    st.session_state.scheduling_trials.append({
        "Trial": len(st.session_state.scheduling_trials) + 1,
        "CO_R": CO_R,
        "MUT_R": MUT_R,
        "Total Fitness": score,
        "Schedule": schedule_df
    })


# Results
if st.session_state.scheduling_trials:

    latest = st.session_state.scheduling_trials[-1]

    st.divider()
    st.subheader("3. Latest Schedule")

    a, b = st.columns(2)

    a.metric(
        "Total Fitness",
        f"{latest['Total Fitness']:.2f}"
    )

    b.metric(
        "Time Slots",
        len(latest["Schedule"])
    )

    st.dataframe(
        latest["Schedule"],
        hide_index=True,
        use_container_width=True
    )

    st.divider()
    st.subheader("4. Experiment History")

    summary = pd.DataFrame([
        {
            key: item[key]
            for key in (
                "Trial",
                "CO_R",
                "MUT_R",
                "Total Fitness"
            )
        }
        for item in st.session_state.scheduling_trials
    ])

    st.dataframe(
        summary,
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
                use_container_width=True
            )

    if st.button("Clear Trial History"):
        st.session_state.scheduling_trials = []
        st.rerun()

else:
    st.info(
        "Upload a CSV if needed, set the parameters, "
        "and click Run Genetic Algorithm."
    )
