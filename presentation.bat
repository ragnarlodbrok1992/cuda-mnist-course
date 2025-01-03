cp presentation\presentation.md cuda-slides\slides.md
cp -r presentation\assets\ cuda-slides\assets\

pushd cuda-slides

npm run dev

popd
