-include ../tools.mk

# Test that -O and -g can be used multiple times, together with their equivalent -C options.

all:
	$(RUSTC) -O -O dummy.rs
	$(RUSTC) -O -C opt-level=2 dummy.rs
	$(RUSTC) -C opt-level=2 -O dummy.rs
	$(RUSTC) -C opt-level=2 -C opt-level=2 dummy.rs

	$(RUSTC) -g -g dummy.rs
	$(RUSTC) -g -C debuginfo=2 dummy.rs
	$(RUSTC) -C debuginfo=2 -g dummy.rs
	$(RUSTC) -C debuginfo=2 -C debuginfo=2 dummy.rs
