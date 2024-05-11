# This test checks that alloc can still compile correctly when the unstable no_global_oom_handling feature is turned on.
# See https://github.com/rust-lang/rust/pull/84266

include ../tools.mk

all:
	$(RUSTC) --edition=2021 -Dwarnings --crate-type=rlib ../../../library/alloc/src/lib.rs --cfg no_global_oom_handling
