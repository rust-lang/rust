# This test verifies that the Rust compiler can still compile correctly when the unstable no_sync feature is turned on, which disables sync in alloc.
# See https://github.com/rust-lang/rust/pull/89891

include ../tools.mk

all:
	$(RUSTC) --edition=2021 -Dwarnings --crate-type=rlib ../../../library/alloc/src/lib.rs --cfg no_sync
