#!/bin/bash

set -ex

../x.py build --stage 1 library

../x.py test --stage 1 tests/codegen-llvm/gpu_offload
