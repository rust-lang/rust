#!/bin/bash

set -ex

../x.py test --stage 2 --no-fail-fast \
    tests/codegen-llvm/autodiff \
    tests/pretty/autodiff \
    tests/ui/autodiff \
    tests/ui/feature-gates/feature-gate-autodiff.rs
