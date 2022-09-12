include ../tools.mk

all:
	$(RUSTC) --edition=2021 -Dwarnings --crate-type=rlib ../../../../library/alloc/src/lib.rs --cfg no_global_oom_handling
