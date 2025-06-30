import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize faker
fake = Faker('es_MX')

# List of products
productos = [
    "3-A Ofteno", "Acquafil Ofteno", "Deltamid", "Dustalox", "Eliptic Ofteno",
    "Gaap Ofteno", "Humylub Ofteno", "Krytantek Ofteno", "Lagricel Ofteno",
    "Manzanilla Sophia", "Meticel Ofteno", "Nazil", "Sopixín DX", "Trazidex"
]

# Generate synthetic dataset
data = []
for _ in range(10):  # sample of 10 records
    dob = fake.date_of_birth(minimum_age=20, maximum_age=80)
    start_treatment = fake.date_this_year(before_today=True, after_today=False)
    continues = random.choice([True, False])
    end_treatment = fake.date_between(start_treatment, fake.date_this_year(before_today=True)) if not continues else None
    onset = fake.date_between(start_treatment, fake.date_this_year(before_today=True))
    end_reaction = fake.date_between(onset, fake.date_this_year(before_today=True))
    record = {
        "Iniciales": "".join([fake.first_name()[0], fake.first_name()[0], fake.last_name()[0]]),
        "Fecha_Nacimiento": dob.strftime("%d/%m/%Y"),
        "Género": random.choice(["M", "F"]),
        "País": fake.country(),
        "Producto": random.choice(productos),
        "Inicio_Tratamiento": start_treatment.strftime("%d/%m/%Y"),
        "Continúa": "Si" if continues else "No",
        "Fin_Tratamiento": end_treatment.strftime("%d/%m/%Y") if end_treatment else "",
        "Lote": fake.bothify(text='LOT-#####'),
        "Evento_Adverso": fake.sentence(nb_words=6),
        "Inicio_Reacción": onset.strftime("%d/%m/%Y"),
        "Fin_Reacción": end_reaction.strftime("%d/%m/%Y"),
        "Descripción": fake.paragraph(nb_sentences=2),
        "Reportero": fake.name(),
        "Relación": random.choice(["Paciente", "Familiar", "Profesional de salud"]),
        "Teléfono": fake.phone_number(),
        "Email": fake.email()
    }
    data.append(record)

df = pd.DataFrame(data)
import ace_tools as tools; tools.display_dataframe_to_user(name="Dataset de Reportes de ADR Sintéticos", dataframe=df)
