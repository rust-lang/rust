include ../../run-make-fulldeps/tools.mk

# How to run this
# $ ./x.py clean
# $ ./x.py test --target thumbv6m-none-eabi,thumbv7m-none-eabi src/test/run-make

# Supported targets:
# - thumbv6m-none-eabi (Bare Cortex-M0, M0+, M1)
# - thumbv7em-none-eabi (Bare Cortex-M4, M7)
# - thumbv7em-none-eabihf (Bare Cortex-M4F, M7F, FPU, hardfloat)
# - thumbv7m-none-eabi (Bare Cortex-M3)

# only-thumb

# For cargo setting
RUSTC := $(RUSTC_ORIGINAL)
LD_LIBRARY_PATH := $(HOST_RPATH_DIR)
# We need to be outside of 'src' dir in order to run cargo
WORK_DIR := $(TMPDIR)

HERE := $(shell pwd)

CRATE := cortex-m
CRATE_URL := https://github.com/rust-embedded/cortex-m
CRATE_SHA1 := a448e9156e2cb1e556e5441fd65426952ef4b927 # 0.5.0

# Don't make lints fatal, but they need to at least warn or they break Cargo's target info parsing.
export RUSTFLAGS := --cap-lints=warn

all:
	env
	mkdir -p $(WORK_DIR)
	-cd $(WORK_DIR) && rm -rf $(CRATE)
	cd $(WORK_DIR) && bash -x $(HERE)/../git_clone_sha1.sh $(CRATE) $(CRATE_URL) $(CRATE_SHA1)
	# HACK(eddyb) sets `RUSTC_BOOTSTRAP=1` so Cargo can accept nightly features.
	# These come from the top-level Rust workspace, that this crate is not a
	# member of, but Cargo tries to load the workspace `Cargo.toml` anyway.
	cd $(WORK_DIR) && cd $(CRATE) && env RUSTC_BOOTSTRAP=1 $(BOOTSTRAP_CARGO) build --target $(TARGET) -v
