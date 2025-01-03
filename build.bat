@echo off
SET compiler_dir="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe"

REM Remember that we are building from "build" directory, so go one below that

IF NOT EXIST build mkdir build

pushd build

%compiler_dir% /EHsc /Zi /MDd^
  /DEBUG:FULL^
  /INCREMENT:NO^
  /std:c++17^
  /Fe:"cuda-mnist"^
  ../src/cuda-mnist.cpp

popd
