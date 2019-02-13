-include ../../run-make-fulldeps/tools.mk

# only-thumbv7m-none-eabi
# only-thumbv6m-none-eabi

# How to run this
# $ ./x.py clean
# $ ./x.py test --target thumbv7m-none-eabi src/test/run-make

# For cargo setting
export RUSTC := $(RUSTC_ORIGINAL)
export LD_LIBRARY_PATH := $(HOST_RPATH_DIR)
# We need to be outside of 'src' dir in order to run cargo
export WORK_DIR := $(TMPDIR)
export HERE := $(shell pwd)

## clean up unused env variables which might cause harm.
unexport RUSTC_LINKER
unexport RUSTC_BOOTSTRAP
unexport RUST_BUILD_STAGE
unexport RUST_TEST_THREADS
unexport RUST_TEST_TMPDIR
unexport AR
unexport CC
unexport CXX

all:
	bash script.sh
