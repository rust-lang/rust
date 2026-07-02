#!/bin/bash

set -ex

../x.py build --stage 1 library

../x.py test --stage 1 --no-fail-fast \
    tests/codegen-llvm/autodiff \
    tests/pretty/autodiff \
    tests/ui/autodiff \
    tests/ui/feature-gates/feature-gate-autodiff.rs
