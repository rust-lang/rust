include ../tools.mk

# FIXME: it would be good to check that it's actually the rightmost flags
# that are used when multiple flags are specified, but I can't think of a
# reliable way to check this.

all:
	# Test that `-O` and `-C opt-level` can be specified multiple times.
	# The rightmost flag will be used over any previous flags.
	$(RUSTC) -O -O main.rs
	$(RUSTC) -O -C opt-level=0 main.rs
	$(RUSTC) -C opt-level=0 -O main.rs
	$(RUSTC) -C opt-level=0 -C opt-level=2 main.rs
	$(RUSTC) -C opt-level=2 -C opt-level=0 main.rs

	# Test that `-g` and `-C debuginfo` can be specified multiple times.
	# The rightmost flag will be used over any previous flags.
	$(RUSTC) -g -g main.rs
	$(RUSTC) -g -C debuginfo=0 main.rs
	$(RUSTC) -C debuginfo=0 -g main.rs
	$(RUSTC) -C debuginfo=0 -C debuginfo=2 main.rs
	$(RUSTC) -C debuginfo=2 -C debuginfo=0 main.rs
