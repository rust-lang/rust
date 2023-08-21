# ignore-cross-compile
include ../tools.mk

# Check that valid binaries are persisted by running them, regardless of whether the --run or --no-run option is used.

MY_SRC_DIR := ${CURDIR}

all: run no_run test_run_directory

run:
	mkdir -p $(TMPDIR)/doctests
	$(RUSTC) --crate-type rlib t.rs
	$(RUSTDOC) -Zunstable-options --test --persist-doctests $(TMPDIR)/doctests --extern t=$(TMPDIR)/libt.rlib t.rs
	$(TMPDIR)/doctests/t_rs_2_0/rust_out
	$(TMPDIR)/doctests/t_rs_8_0/rust_out
	rm -rf $(TMPDIR)/doctests

no_run:
	mkdir -p $(TMPDIR)/doctests
	$(RUSTC) --crate-type rlib t.rs
	$(RUSTDOC) -Zunstable-options --test --persist-doctests $(TMPDIR)/doctests --extern t=$(TMPDIR)/libt.rlib t.rs --no-run
	$(TMPDIR)/doctests/t_rs_2_0/rust_out
	$(TMPDIR)/doctests/t_rs_8_0/rust_out
	rm -rf $(TMPDIR)/doctests

# Behavior with --test-run-directory with relative paths.
test_run_directory:
	mkdir -p $(TMPDIR)/doctests
	mkdir -p $(TMPDIR)/rundir
	$(RUSTC) --crate-type rlib t.rs
	( cd $(TMPDIR); \
		$(RUSTDOC) -Zunstable-options --test --persist-doctests doctests --test-run-directory rundir --extern t=libt.rlib $(MY_SRC_DIR)/t.rs )
	rm -rf $(TMPDIR)/doctests $(TMPDIR)/rundir
