@echo off

set DIRECTORY="press_build"
set PRESENTATION_FILE="../presentation/presentation.md"
set OUTPUT_FILE="index.html"

IF NOT EXIST %DIRECTORY% MKDIR %DIRECTORY%

cp -r ./presentation/assets/ %DIRECTORY%/assets/

pushd %DIRECTORY%

pandoc -s --mathjax -i -t revealjs --variable theme=moon --variable transition=fade %PRESENTATION_FILE% -o %OUTPUT_FILE%

popd

start python -m http.server 8080 --directory %DIRECTORY%/

