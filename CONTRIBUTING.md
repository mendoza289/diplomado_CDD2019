# Como contribuir al Dipomado en Ciencia de Datos 2019
## 1. Descarga el repositorio
- Instala Git para cualquier SO: [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Descarga o clona en tu directorio de trabajo el repositorio; puedes consultar esta [guía rápida y fácil](https://rogerdudler.github.io/git-guide/). 
Esto crea en tu directorio de trabajo una copia de todos los archivos del repositorio. La idea es que ahora cada colaborador debe 
crear una rama de trabajo con su propio Módulo del diplomado. 

## 2. Conocer el proceso de trabajo (Work Flow) de Git (ver por ejemplo esta [guía](https://rogerdudler.github.io/git-guide/)):
- actualizar su repositorio local con [`git pull origin master`](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/),
- crear una rama de trabajo con [`git checkout -b MyNewBranch`](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/)
- realizar los cambios en su rama y prepararlos con [`git add`](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/),
- confirmar tus cambios localmente con [`git commit -m "descripción de tu confirmación"`](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/), y
- subir los cambios (incluida su nueva rama) en GitHub con [`git push origin MyNewBranch`](https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/)
- Vaya al repositorio principal en GitHub donde ahora debería ver su nueva rama
- haga clic en el nombre de su rama
- haga clic en el botón "Pull Request"
- haga clic en "Send Pull Request"

## 1. Conocer la [terminología de Git](https://rogerdudler.github.io/git-guide/):
### REPOS AND BRANCHES
| Term            | Description                                                                                                                                                    |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Origin (repo)   | Your remote repo (on Github); it is the “origin” for your local copy.  Either it is a repo you created yourself or it is a fork of someone else’s GitHub repo. |
| Upstream (repo) | The main repo (on GitHub) from which you forked your GiHub repo.                                                                                               |
| Local (repo)    | The repo on your local computer.                                                                                                                               |
| Master          | The main branch (version) of your repo.                                                                                                                        |

### BASIC COMMANDS/ACTIONS
| Term         | Explanation                                                                                                                                                                   |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fork         | Make a copy of someone else’s GitHub repo in your own GitHub account.                                                                                                         |
| Clone        | Make a copy of the your GitHub repo on your local computer. In CLI (Computer Local Interface): ‘git clone’ copies a remote repo to create a local repo with a remote called origin automatically set up. |
| Pull         | You incorporate changes into your repo.                                                                                                                                       |
| Add          | Adding snapshots of your changes to the “Staging” area.                                                                                                                       |
| Commit       | Takes the files as they are in your staging area and stores a snap shot of your files (changes) permanently in your Git directory.                                            |
| Push         | You “push” your files (changes) to the remote repo.                                                                                                                           |
| Merge        | Incorporate changes into the branch you are on.                                                                                                                               |
| Pull Request | Term used in collaboration. You “issue a pull request” to the owner of the upstream repo asking them to pull your changes into their repo (accept your work).                 |
