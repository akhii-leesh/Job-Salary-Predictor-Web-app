import pickle
import numpy as np
import streamlit as st

m = dict()
with open('saved.pkl','rb') as f:
    m = pickle.load(f)

model = m['model']
le_size = m['le_size']
le_own = m['le_own']
le_ind = m['le_ind']
le_sect = m['le_sect']
le_revenue = m['le_revenue']
le_job = m['le_job']
le_seniority = m['le_seniority']


def show_page():
    st.title("Job Salary Predictor")
    st.write("Enter the required information to predict the salary of the specified job:")
    company_size = {
        "1001 to 5000 employees",
        "501 to 1000 employees",
        "10000+ employees",
        "201 to 500 employees",
        "51 to 200 employees",
        "5001 to 10000 employees",
        "1 to 50 employees",
        "Unknown"
    }
    ownership_type = {
        "Company - Private",
        "Company - Public",
        "Nonprofit Organization",
        "Subsidiary or Business Segment",
        "Government",
        "Hospital",
        "College / University",
        "Other Organization",
        "Unknown",
        "School / School District"
    }
    industry = {
        "Biotech & Pharmaceuticals",
        "Insurance Carriers",
        "Computer Hardware & Software",
        "IT Services",
        "Health Care Services & Hospitals",
        "Enterprise Software & Network Solutions",
        "Internet",
        "Consulting",
        "Aerospace & Defense",
        "Advertising & Marketing",
        "Consumer Products Manufacturing",
        "Research & Development",
        "Colleges & Universities",
        "Energy",
        "Banks & Credit Unions"
    }
    sector = {
        "Information Technology",
        "Biotech & Pharmaceuticals",
        "Business Services",
        "Insurance",
        "Health Care",
        "Finance",
        "Manufacturing",
        "Aerospace & Defense",
        "Education",
        "Retail",
        "Oil, Gas, Energy & Utilities",
        "Government",
        "Unknown",
        "Non-Profit",
        "Transportation & Logistics",
        "Real Estate",
        "Travel & Tourism",
        "Telecommunications",
        "Media"
    }
    revenue = {
        "Unknown / Non-Applicable",
        "$10+ billion (USD)",
        "$100 to $500 million (USD)",
        "$1 to $2 billion (USD)",
        "$500 million to $1 billion (USD)",
        "$50 to $100 million (USD)",
        "$25 to $50 million (USD)",
        "$2 to $5 billion (USD)",
        "$10 to $25 million (USD)",
        "$5 to $10 billion (USD)",
        "$5 to $10 million (USD)",
        "$1 to $5 million (USD)",
        "Less than $1 million (USD)"
    }
    job_spec = {
        "data scientist",
        "other",
        "data engineer",
        "analyst",
        "manager",
        "mle",
        "director"
    }
    seniority = {
        "na",
        "sr",
        "jr"
    }

    comp = st.selectbox("Company size / no. of employees", company_size)
    own = st.selectbox("Ownership type", ownership_type)
    ind = st.selectbox("Industry", industry)
    sect = st.selectbox("Sector", sector)
    rev = st.selectbox("Revenue", revenue)
    job = st.selectbox("Job specification", job_spec)
    sen = st.selectbox("Seniority", seniority)
    compage = st.slider("Company age", 0,50,3)
    rating = st.slider("Company rating", 0,5,1)

    ok = st.button("Calculate salary")

    if ok:
        X = np.array([[rating,comp,own,ind,sect,rev,job,sen,3000,compage,1,1,1,0,0,1,1,1]])
        X[:,1] = le_size.transform(X[:,1])
        X[:,2] = le_own.transform(X[:,2])
        X[:,3] = le_ind.transform(X[:,3])
        X[:,4] = le_sect.transform(X[:,4])
        X[:,5] = le_revenue.transform(X[:,5])
        X[:,6] = le_job.transform(X[:,6])
        X[:,7] = le_seniority.transform(X[:,7])

        X.astype(float)
        salary = model.predict(X)
        st.subheader(f"The estimated salary is ${int(salary[0])} K per annum")

show_page()