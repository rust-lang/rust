-include ../tools.mk

# Ensure that rustdoc doesn't erroneously space-split arguments given to --test-args, thus allowing
# users to properly filter out tests by name and line number

all:
	$(RUSTDOC) lib.rs --test --test-args "(line 3)"
	$(RUSTDOC) lib.rs --test --test-args "(line 3)" --test-args "--list" | $(CGREP) '1 test, 0 benchmarks'
