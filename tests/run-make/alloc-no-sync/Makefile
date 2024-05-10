# This test checks that alloc can still compile correctly when the unstable no_sync feature is turned on.
# See https://github.com/rust-lang/rust/pull/89891

include ../tools.mk

all:
	$(RUSTC) --edition=2021 -Dwarnings --crate-type=rlib ../../../library/alloc/src/lib.rs --cfg no_sync
