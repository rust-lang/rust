include ../tools.mk

FAKEROOT=$(TMPDIR)/fakeroot

all:
	$(RUSTC) --edition=2021 -Dwarnings --crate-type=rlib ../../../library/core/src/lib.rs --sysroot=$(FAKEROOT) --cfg no_global_oom_handling
