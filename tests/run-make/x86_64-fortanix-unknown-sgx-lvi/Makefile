include ../../run-make-fulldeps/tools.mk

#only-x86_64-fortanix-unknown-sgx

# For cargo setting
export RUSTC := $(RUSTC_ORIGINAL)
export LD_LIBRARY_PATH := $(HOST_RPATH_DIR)
# We need to be outside of 'src' dir in order to run cargo
export WORK_DIR := $(TMPDIR)
export TEST_DIR := $(shell pwd)

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
