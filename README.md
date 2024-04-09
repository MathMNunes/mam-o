# Avaliação da qualidade com inteligência artificial em mamões 

---
#### Projeto da Disciplina PET1706 - TÓPICOS ESPECIAIS EM ENGENHARIA DE SOFTWARE (Redes Neurais Artificiais) - 2023.2 
###### Professora: [Rosana Rego](https://github.com/roscibely)
<div>
  <img src="https://raw.githubusercontent.com/roscibely/algorithms-and-data-structure/main/root/ufersa.jpg" width="700" height="250">
</div>
<i> <a href="https://engsoftwarepaudosferros.ufersa.edu.br/apresentacao/"> Curso Bacharel em Engenharia de Software  </a> - UFERSA - Campus Pau dos Ferros </a></i>

---



## Funcionalidades Principais

- **Modelo de Machine Learning**: O sistema é capaz de analisar imagens e determinar seu nível de maturidade.O modelo foi treinado previamente com imagens de mamões com diferentes níveis de maturidade, Implicando que foi alimentado com um conjunto de dados e treinado para reconhecer e classificar a imagem com base nas suas caracteristicas.

- **Servidor Web**: O sistema é construído utilizando a ferramenta Django. Django é um framework web em Python que facilita a construção de aplicações web robustas e escaláveis. O sistema gerencia rotas podendo lidar com solicitações HTTP, disponibilizado link principal com opção de retornar o resultado em formato json (API), acessar bancos de dados, e fornecer uma interface web para os usuários interagirem com a aplicação.

## Requisitos de Sistema

- Python (versão 3.10.9)
- pip (gerenciador de pacotes do Python)
- Virtualenv (opcional, mas altamente recomendado)

## Instalação

Siga estas etapas para configurar e executar o projeto localmente.

```bash
# Clone o repositório:
https://github.com/ClassNeuralNetwork/maturidade_mamoes

# Navegue até o diretório do projeto:
cd papaya_quality

# Crie e ative um ambiente virtual (opcional, mas recomendado):
virtualenv venv
source venv/bin/activate

# Instale as dependências do projeto:
pip install -r requirements.txt

# Configure as variáveis de ambiente:
# Crie um arquivo `.env` na raiz do projeto e defina as variáveis de ambiente necessárias, como chaves secretas, configurações de banco de dados, etc.
# As variaveis de desenvolvimento estão no arquivo .env_auxiliar, pode copiar e colar dentro do arquivo .env

# Aplique as migrações do Django:
python manage.py migrate

# Crie um superusuário (opcional):
python manage.py createsuperuser
```

## Uso
### Executando o Servidor de Desenvolvimento
### Execute o servidor de desenvolvimento com o seguinte comando:

```bash
python manage.py runserver
```
Acesse o projeto em seu navegador em http://localhost:8000.

## Contribuição

Contribuições são bem-vindas! Se você quiser contribuir para este projeto, por favor, abra uma issue para discutir as mudanças propostas ou envie um pull request.
## Equipe
<table align="center">
  <tr>    
    <td align="center">
      <a href="https://github.com/cristiana0">
        <img src="https://avatars.githubusercontent.com/u/85590409?v=4" 
        width="120px;"  alt="Foto de Cristiana Paulo no GitHub"/><br>
        <sub>
          <b>Cristiana Paulo</b>
         </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Emanuel-Bruno">
        <img src="https://avatars.githubusercontent.com/u/58535705?v=4" 
        width="120px;" alt="Foto de Emanuel Morais no GitHub"/><br>
        <sub>
          <b>Emanuel Morais</b>
         </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/LucasMatheus12">
        <img src="https://avatars.githubusercontent.com/u/96743905?v=4" 
        width="120px;" alt="Foto de Lucas Matheus no GitHub"/><br>
        <sub>
          <b>Lucas Matheus</b>
         </sub>
      </a>
    </td>
  </tr>
</table>

<p align="center">
Cada contribuidor desempenhou um papel essencial no desenvolvimento e aprimoramento deste projeto.
</p>



## Licença

Este projeto está licenciado sob a [Licença MIT](https://opensource.org/licenses/MIT). Consulte o arquivo `LICENSE` para obter mais detalhes.

## Agradecimentos

Agradecemos aos desenvolvedores, à comunidade de código aberto com as ferramentes abertas que utilizamos para construção deste projeto, a comunidade de Machine Learning, ao professor Dr. Nildo da Silva Dias pela disponibilização do dataset com as imagens.
