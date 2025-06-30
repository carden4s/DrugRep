import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

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
hoy = datetime.now().date()

for _ in range(10):  # sample of 10 records
    # Fecha de nacimiento (entre 20 y 80 años)
    dob = fake.date_of_birth(minimum_age=20, maximum_age=80)
    
    # Inicio de tratamiento: en el año actual, antes de hoy
    start_treatment = fake.date_this_year(before_today=True, after_today=False)
    
    # Determina si continúa el tratamiento
    continues_treatment = random.choice([True, False])
    if continues_treatment:
        end_treatment = None
    else:
        # Fin de tratamiento: entre el inicio y hoy
        end_treatment = fake.date_between(start_date=start_treatment, end_date=hoy)
    
    # Generar iniciales consistentes
    nombre = fake.first_name()
    apellido = fake.last_name()
    partes_nombre = nombre.split()
    
    # Manejar nombres simples y compuestos
    if len(partes_nombre) >= 2:
        inicial1 = partes_nombre[0][0]
        inicial2 = partes_nombre[1][0]
    else:
        if len(partes_nombre[0]) > 1:
            inicial1 = partes_nombre[0][0]
            inicial2 = partes_nombre[0][1]
        else:
            inicial1 = partes_nombre[0][0]
            inicial2 = partes_nombre[0][0]
    iniciales = f"{inicial1}{inicial2}{apellido[0]}"
    
    # Fecha de inicio de reacción: puede ser antes, durante o después del tratamiento
    # (30% de probabilidad de que comience antes del tratamiento)
    if random.random() < 0.3:
        fecha_min = start_treatment - timedelta(days=30)
        fecha_max = start_treatment
    else:
        fecha_min = start_treatment
        fecha_max = min(start_treatment + timedelta(days=90), hoy)
    
    onset = fake.date_between_dates(date_start=fecha_min, date_end=fecha_max)
    
    # Determinar si la reacción continúa
    reaccion_continua = random.choice([True, False])
    if reaccion_continua:
        end_reaction = None
    else:
        # Fin de reacción: entre el inicio de la reacción y hoy
        end_reaction = fake.date_between_dates(date_start=onset, date_end=hoy)

    record = {
        "Iniciales": iniciales,
        "Fecha_Nacimiento": dob.strftime("%d/%m/%Y"),
        "Género": random.choice(["M", "F"]),
        "País": fake.country_code(representation="alpha-2"),  # código de 2 letras
        "Producto": random.choice(productos),
        "Inicio_Tratamiento": start_treatment.strftime("%d/%m/%Y"),
        "Continúa": "Si" if continues_treatment else "No",
        "Fin_Tratamiento": end_treatment.strftime("%d/%m/%Y") if end_treatment else "",
        "Lote": fake.bothify(text='LOT-??###'),  # Formato alfanumérico
        "Evento_Adverso": fake.sentence(nb_words=6),
        "Inicio_Reacción": onset.strftime("%d/%m/%Y"),
        "Fin_Reacción": end_reaction.strftime("%d/%m/%Y") if end_reaction else "",
        "Descripción": fake.paragraph(nb_sentences=2),
        "Reportero": fake.name(),
        "Relación": random.choice(["Paciente", "Familiar", "Profesional de salud"]),
        "Teléfono": fake.phone_number(),
        "Email": fake.email()
    }
    data.append(record)

df = pd.DataFrame(data)
import ace_tools as tools; tools.display_dataframe_to_user(name="Dataset de Reportes de ADR Sintéticos", dataframe=df)