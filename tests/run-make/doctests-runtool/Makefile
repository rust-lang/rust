# ignore-cross-compile
include ../tools.mk

# Tests behavior of rustdoc --runtool

MY_SRC_DIR := ${CURDIR}

all: with_test_run_directory

# Behavior with --runtool with relative paths and --test-run-directory.
with_test_run_directory:
	mkdir -p $(TMPDIR)/rundir
	mkdir -p $(TMPDIR)/runtool
	$(RUSTC) --crate-type rlib t.rs
	$(RUSTC) runtool.rs -o $(TMPDIR)/runtool/runtool
	( cd $(TMPDIR); \
		$(RUSTDOC) -Zunstable-options --test --test-run-directory rundir \
			--runtool runtool/runtool --extern t=libt.rlib $(MY_SRC_DIR)/t.rs \
	)
	rm -rf $(TMPDIR)/rundir $(TMPDIR)/runtool
