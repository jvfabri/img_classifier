 - Identificação da Presença de Aneis em Imagens de Blocos de Motor 
	
	Autor: João Victor Fabri
	e-mail: joaovictor.fabri@gmail.com

REQUISITOS/BIBLIOTECAS NECESSÁRIAS:

	Python 3
	Tensorflow 2.x e submódulos necessários (Keras, Pandas, etc.)
	Numpy
	MatPlotLib

ORGANIZAÇÃO SUGERIDA PARA AS PASTAS COM AS IMAGENS:

	root:
	|-training.py
	|-evaluation.py
	|-saved_model.h5 ** 
	|-test:
		|-sem_anel:
			|-img.bmp
			|...
		|-com_anel:
			|-img.bmp
			|...
	|-train:
		|-sem_anel:
			|-img.bmp
			|...
		|-com_anel:
			|-img.bmp
			|...
	|-predict:
		|-imagem_sem_anel.bmp***
	**arquivo criado após a execução de "training.py"
	***opcional

CORTA CAMINHO:

	>>python3 training.py train 
	--- aguarde resultado do treinamento

	>>python3 evaluation.py 
	--- executa avaliação do modelo sem classificar qualquer imagem. 
	OU
	>>python3 evaluation.py predict/imagem_sem_anel.bmp
	--- executa avaliação do modelo e classifica a imagem "imagem_sem_anel" entre Sem ou Com anel.
	OU
	>>python3 evaluation.py predict/imagem_sem_anel.bmp --notest
	--- executa a classificação da imagem "imagem_sem_anel" sem realizar a validação do modelo.

COMO EXECUTAR:

- Organize os dados de treinamento e validação em subpastas de acordo com sua classificação ("Sem anel" ou "Com anel").

- Treinamento da rede:
		Execute o script "training.py" a partir de uma pasta superior aos dados de treinamento a partir do comando:

	>> python3 training.py arg1 
	
		onde arg1 é o caminho relativo para a pasta que contem as subpastas com as imagens de treinamento. 
		Caso arg1 estiver em branco, o script utilizará as imagens das pastas que estiverem no mesmo diretório de "training.py"
		
	Exemplo: 
	>> python3 training.py train
		
		O algoritmo buscará as imagens das pastas "sem_anel" e "com_anel" dentro da pasta "train", na mesma pasta do 
		script.
		Caso o diretório selecionado tenha mais do que 2 pastas, o programa acusará erro de compilação.
		Caso seja bem sucedido, iniciará o processo de treinamento, identificando o número de épocas, os passos de 
		treinamento concluídos, o tempo estimado por época e por passo, o erro de treinamento e a acurácia da rede, 
		como no exemplo a seguir:
	
	Epoch 14/50
	100/100 [==============================] - XXs XXXXms/step - loss: X.XXXX - accuracy: X.XXXX

		Após a finalização da sessão de treinamento, será impresso no terminal o tempo total de treinamento e uma 
		mensagem de finalização do processo:

	Elapsed time: XXXX.XXXXXX s
	done! -- run evaluation.py to evaluate the model or classify new images

		Note que uma sessão completa de treinamento não dura necessariamente 50 épocas, tendo em vista que o modelo 
		identifica estagnação no aprendizado e finaliza a sessão de treinamento. 
		Ao fim do treinamento, um arquivo é gerado ("saved_model.h5"), que será utilizado pelo script de avaliação 
		do modelo.

- Avaliação da rede:

	 	Execute o script "evaluation.py" a partir de uma pasta superior aos dados de avaliação a partir do comando:
	
	>> python3 evaluation.py arg1 arg2
	
		onde arg1 pode conter o caminho relativo para uma imagem em ".bmp" a ser classificada. 
		e onde arg2 pode ou não conter a flag "--notest", que faz com que não seja executado o passo de validação, 
		permitindo a classificação mais rápida de imagens pelo usuário.
		
		Para que a validação possa ser executada corretamente, as imagens de validação devem ser colocadas em uma 
		pasta chamada test no mesmo diretório do script "evaluation.py"

	Exemplo:
	>>python3 evaluation.py predict/imagem_sem_anel.bmp 
	
		Executa a validação do modelo e a classificação da imagem "imagem_sem_anel.bmp", dentro da pasta predict.
		Isso resulta na impressão de informações sobre a validação do modelo, com o número de images e classes 
		utilizadas na validação, as métricas de erro (Sparse Categorical Crossentropy e acurácia) e o tempo do 
		processamento da validação, como mostrado a seguir:

	Found XX images belonging to 2 classes. 
	SCC (Loss): X.XXXXXXXXXE+XX, Accuracy: X.XXX, Elapsed Time: X.XXXXXXXXXX s
	Sem anel

		Caso a validação não seja executada, apenas a primeira e terceira linha do exemplo serão impressas na tela.
		É gerada uma matriz de imagens com a identificação das características (features) da imagem classificada, 
		para melhor visualização do funcionamento da rede. Essa imagem é salva com o nome de 
		"features_nomedoarquivo_bmp.pdf" na mesma pasta do script de avaliação. 
	
		
		
		
