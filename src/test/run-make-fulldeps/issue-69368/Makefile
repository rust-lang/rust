include ../tools.mk

# Test that previously triggered a linker failure with root cause
# similar to one found in the issue #69368.
#
# The crate that provides oom lang item is missing some other lang
# items. Necessary to prevent the use of start-group / end-group.
#
# The weak lang items are defined in a separate compilation units,
# so that linker could omit them if not used.
#
# The crates that need those weak lang items are dependencies of
# crates that provide them.

all:
	$(RUSTC) a.rs
	$(RUSTC) b.rs
	$(RUSTC) c.rs
