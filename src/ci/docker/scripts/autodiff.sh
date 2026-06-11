#!/bin/bash

set -ex

../x.py build --stage 1 library

../x.py test --stage 1 tests/codegen-llvm/autodiff
../x.py test --stage 1 tests/pretty/autodiff
../x.py test --stage 1 tests/ui/autodiff
../x.py test --stage 1 tests/run-make/autodiff
../x.py test --stage 1 tests/ui/feature-gates/feature-gate-autodiff.rs
