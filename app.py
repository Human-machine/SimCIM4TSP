import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np 
import torch
import matplotlib.pyplot as plt
from SimCIM import*
from functions import*
datatype = torch.float32
device = 'cpu'

url = "https://www.science.org/doi/10.1126/science.aah4243"
st.write("""
# SimCIM for TSP
 The app for finding TSP solutions by simulating the [Coherent Ising Machine](%s) (CIM)
""" % url)

# st.write("check out this [link](%s)" % url)
# st.markdown("check out this [link](%s)" % url)


st.write('''Here I present the algorithm called SimCIM for solving Travelling Salesman Problem,
	i. e. finding the route passing through each city once with the least distance.''')
st.write('''For more information about SimCIM, please read the paper''')

url = 'https://doi.org/10.1364/OE.27.010288'
st.markdown(url, unsafe_allow_html=True)

st.write('''The solution quality depends on the choice of three hyperparameters O, S, D. 
	You need to adjust them to find best possible route.''')

st.sidebar.header("User input parameters")

def user_input_parameters():
	params = {}
	params['O'] = st.sidebar.text_input('O',value = 0.05)
	params['D'] = st.sidebar.text_input('D', value = -0.8)
	params['S'] = st.sidebar.text_input('S', value = 0.1)
	data = {'O': params['O'],
			'S': params['S'],
			'D': params['D']}
	features = pd.DataFrame(data,  index = [0])
	return features,params

df,params = user_input_parameters()

st.subheader("User input parameters")
st.write(df)

def run_simcim(cities,lengths):
	fig = plot_cities(cities,lengths,np.random.permutation(cities.shape[0]))
	st.pyplot(fig)
	B = 0.1
	A = 1.5*B*lengths.max()
	J,b = get_Jh(lengths, A, B)

	simcim = Simcim(J,b,device,datatype)
	for key in params.keys():
		simcim.params_cont[key] = float(params[key])

	c_current, c_evol = simcim.evolve()
	s_cur = torch.sign(c_current)
	E = energy(J,b,s_cur)

	st.write('Evolution of amplitudes')
	fig, ax = plt.subplots(figsize = (5,5))
	for i in range(J.size(0)):
		ax.plot(c_evol[i,:].cpu().numpy())
	ax.grid()
	ax.set_xlabel('time step')
	ax.set_ylabel('amplitudes')
	fig.tight_layout()
	st.pyplot(fig)

	s_min = s_cur[:,torch.argmin(E)]
	order = get_order_simcim(s_min,N_cities)
	fig = plot_cities(cities,lengths,order)
	st.write('Best route among '+str(simcim.params_disc['attempt_num'])+' runs')
	st.pyplot(fig)

st.subheader('Upload file containing city coordinates')
st.write('File has to be a 2D array of the shape = (Number of cites, 2)')
st.write('Each row of the file denotes x and y coordinates of a particular city')
uploaded_cities = st.file_uploader("Choose a file")
if st.button('Run SimCIM based on uploaded file information') and (uploaded_cities is not None):
	stringio = StringIO(uploaded_cities.getvalue().decode("utf-8"))
	string_data = stringio.read()
	cities = np.fromstring(string_data, sep = ' ')
	N_cities = int(len(cities)/2)
	cities = cities.reshape(N_cities,2)
	L = (cities.reshape(cities.shape[0],1,2) - cities.reshape(1,cities.shape[0],2))**2
	lengths = np.sqrt(np.sum(L,2))
	run_simcim(cities,lengths)


st.subheader('Generate coordinates of N cities randomly')
N_cities = st.text_input('Number of cities', value = 20)
N_cities = int(N_cities)
if st.button('Generate cities randomly and run SimCIM'):
	cities,lengths = generate_cities(N_cities)
	run_simcim(cities,lengths)
	










