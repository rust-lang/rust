-include ../tools.mk

all:
	$(RUSTC) --edition=2018 --crate-type=rlib ../../../../library/alloc/src/lib.rs --cfg feature=\"external_crate\" --cfg no_global_oom_handling
