from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
from math import log10


app = Flask(__name__)
# LOGO_PATH = 'static/images/PennMedicine.jpg'
BACKGROUND_PATH = 'static/images/background.png'
LOGO_PATH = None
# BACKGROUND_PATH = None

# @app.route("/")
# def hello_world():
#     return f"<p>Hello, World!</p>"

# @app.route("/")
# def index():
#     return render_template('index.html')


@app.route("/")
def index():
    return render_template('index.html', cycloplegic_prediction_left="None", myopia_prediction_left="None",
                           cycloplegic_prediction_right="None", myopia_prediction_right="None", image_path=LOGO_PATH,
                           background_path=BACKGROUND_PATH)


@app.route("/model", methods=['GET', 'POST'])
def model():
    ml_model = request.form.get('models')
    age = request.form.get('age')
    gender = request.form.get('gender')

    kwargs = {
        'model': ml_model,
        'age': age,
        'gender': gender,
        'image_path': LOGO_PATH,
        'background_path': BACKGROUND_PATH,
    }

    vars = [
        'glasses',
        'sphe',
        'sphere',
        'cylinder',
        'ucva_snellen',
        'ucva_logmar',
        'al',
        'cr',
        'al_cr_ratio',
        'iop',
    ]

    cycloplegic_model_file_path = f'models/cycloplegic_spherical_equivalent/{ml_model}_pipeline.pkl'
    myopia_model_file_path = f'models/myopia_status/{ml_model}_pipeline.pkl'

    for suffix in ['left', 'right']:

        for var in vars:
            exec(f'{var}_{suffix} = request.form.get("{var}_{suffix}")')
            kwargs[f'{var}_{suffix}'] = eval(f'{var}_{suffix}')

        exec(f'sphe_error_{suffix} = sphe_{suffix} and (sphere_{suffix} or cylinder_{suffix})')
        exec(f'sphe_req_{suffix} = sphe_{suffix} or (sphere_{suffix} and cylinder_{suffix})')
        kwargs[f'sphe_error_{suffix}'] = eval(f'sphe_error_{suffix}')
        kwargs[f'sphe_req_{suffix}'] = eval(f'sphe_req_{suffix}')

        exec(f'ucva_error_{suffix} = ucva_snellen_{suffix} and ucva_logmar_{suffix}')
        exec(f'ucva_req_{suffix} = ucva_snellen_{suffix} or ucva_logmar_{suffix}')
        kwargs[f'ucva_error_{suffix}'] = eval(f'ucva_error_{suffix}')
        kwargs[f'ucva_req_{suffix}'] = eval(f'ucva_req_{suffix}')

        exec(f'al_cr_error_{suffix} = al_cr_ratio_{suffix} and (al_{suffix} or cr_{suffix})')
        exec(f'al_cr_req_{suffix} = al_cr_ratio_{suffix} or (al_{suffix} and cr_{suffix})')
        kwargs[f'al_cr_error_{suffix}'] = eval(f'al_cr_error_{suffix}')
        kwargs[f'al_cr_req_{suffix}'] = eval(f'al_cr_req_{suffix}')

        exec(f'unfilled_error_{suffix} = '
             f'not (ml_model and age and gender and glasses_{suffix} and sphe_req_{suffix} '
             f'and ucva_req_{suffix} and al_cr_req_{suffix} and iop_{suffix})')
        exec(f'input_error_{suffix} = '
             f'sphe_error_{suffix} or ucva_error_{suffix} or al_cr_error_{suffix} or unfilled_error_{suffix}')

        kwargs[f'unfilled_error_{suffix}'] = eval(f'unfilled_error_{suffix}')
        kwargs[f'input_error_{suffix}'] = eval(f'input_error_{suffix}')

        if eval(f'input_error_{suffix}'):
            kwargs[f'cycloplegic_prediction_{suffix}'] = "None"
            kwargs[f'myopia_prediction_{suffix}'] = "None"
            continue

        exec(f'app_sphe_{suffix} = get_ser(sphere_{suffix}, cylinder_{suffix}) if not sphe_{suffix} else sphe_{suffix}')
        exec(f'app_al_cr_ratio_{suffix} = '
             f'get_al_cr_ratio(al_{suffix}, cr_{suffix}) if not al_cr_ratio_{suffix} else al_cr_ratio_{suffix}')
        exec(f'app_ucva_snellen_{suffix} = '
             f'get_ucva_snellen(ucva_logmar_{suffix}) if ucva_logmar_{suffix} else ucva_snellen_{suffix}')
        exec(f'app_ucva_logmar_{suffix} = '
             f'get_ucva_logmar(ucva_snellen_{suffix}) if ucva_snellen_{suffix} else ucva_logmar_{suffix}')

        kwargs[f'app_sphe_{suffix}'] = eval(f'app_sphe_{suffix}')
        kwargs[f'app_al_cr_ratio_{suffix}'] = eval(f'app_al_cr_ratio_{suffix}')
        kwargs[f'app_ucva_snellen_{suffix}'] = eval(f'app_ucva_snellen_{suffix}')
        kwargs[f'app_ucva_logmar_{suffix}'] = eval(f'app_ucva_logmar_{suffix}')

        exec(f'cycloplegic_result_{suffix} = '
             f'model_output(model_file_path=cycloplegic_model_file_path, age=age, gender=gender, '
             f'glasses=glasses_{suffix}, sphe=app_sphe_{suffix}, ucva=app_ucva_logmar_{suffix}, '
             f'al_cr_ratio=app_al_cr_ratio_{suffix}, iop=iop_{suffix})'
             )
        exec(f"cycloplegic_prediction_{suffix} = round(cycloplegic_result_{suffix}['prediction'], 2)")
        if eval(f"cycloplegic_prediction_{suffix}") > 0:
            exec(f"cycloplegic_prediction_{suffix} = '+' + str(cycloplegic_prediction_{suffix})")

        exec(f'myopia_result_{suffix} = '
             f'model_output(model_file_path=myopia_model_file_path, age=age, gender=gender, '
             f'glasses=glasses_{suffix}, sphe=app_sphe_{suffix}, ucva=app_ucva_logmar_{suffix}, '
             f'al_cr_ratio=app_al_cr_ratio_{suffix}, iop=iop_{suffix})'
             )

        if eval(f'myopia_result_{suffix}') == "None":
            exec(f"myopia_prediction_{suffix}='None'")
            exec(f"myopia_prediction_probability_{suffix}='None'")

        else:
            exec(f"myopia_prediction_{suffix}=myopia_result_{suffix}['prediction']")
            exec(f"myopia_prediction_{suffix}='Myopic' if myopia_prediction_{suffix} == 1 else 'Not Myopic' if myopia_prediction_{suffix} == 0 else 'None'")
            exec(f"myopia_prediction_probability_{suffix}=round(myopia_result_{suffix}['prediction_probability'], 4)")

        kwargs[f'cycloplegic_prediction_{suffix}'] = eval(f'cycloplegic_prediction_{suffix}')
        kwargs[f'myopia_prediction_{suffix}'] = eval(f'myopia_prediction_{suffix}')
        kwargs[f'myopia_prediction_probability_{suffix}'] = eval(f'myopia_prediction_probability_{suffix}')

    # if input_error:
    #     return render_template('index.html', model=ml_model, age=age, gender=gender, glasses=glasses, sphe=sphe,
    #                            sphere=sphere, cylinder=cylinder, ucva_snellen=ucva_snellen, ucva_logmar=ucva_logmar,
    #                            al=al, cr=cr, al_cr_ratio=al_cr_ratio, iop=iop, input_error=input_error,
    #                            sphe_error=sphe_error, ucva_error=ucva_error, al_cr_error=al_cr_error,
    #                            unfilled_error=unfilled_error,cycloplegic_prediction="None", myopia_prediction="None",
    #                            image_path=LOGO_PATH, background_path=BACKGROUND_PATH)

    return render_template('index.html', **kwargs)


def model_output(model_file_path, age, gender, glasses, sphe, ucva, al_cr_ratio, iop):
    if not os.path.exists(model_file_path):
        return "None"

    with open(model_file_path, 'rb') as file:
        model_pipeline = pickle.load(file)

    gender = 1 if gender == "male" else 2 if gender == "female" else None
    glasses = 1 if glasses == "yes" else 0 if glasses == "no" else None

    x_df = pd.DataFrame({'GENDER': [gender], 'glasses': [glasses], 'age': [float(age)], 'sphe': [float(sphe)],
                         'UCVA_logMAR': [float(ucva)],
                         'AL_CR_ratio': [float(al_cr_ratio)], 'IOP': [float(iop)]})
    x_to_transform = x_df.copy(deep=True)

    for transformer_name, transformer in model_pipeline.steps[:-1]:
        x_to_transform = transformer.transform(x_to_transform)
    x = x_to_transform

    estimator = model_pipeline['estimator']
    result = estimator.predict(x)[0]
    output_dict = {'prediction': result}

    if hasattr(estimator, "predict_proba"):
        predicted_proba = estimator.predict_proba(x)[:, 1][0]
        output_dict['prediction_probability'] = predicted_proba

    return output_dict


def get_ser(sphere, cylinder):
    return float(sphere) + float(cylinder) / 2


def get_al_cr_ratio(al, cr):
    return float(al) / float(cr)


def get_ucva_snellen(ucva_logmar):
    return round(20 * 10 ** float(ucva_logmar))


def get_ucva_logmar(ucva_snellen):
    if float(ucva_snellen) == 20:
        return 0  # returning -0 when 20 is input
    return -log10(20/float(ucva_snellen))

if __name__ == '__main__':
    app.run(debug=True)
