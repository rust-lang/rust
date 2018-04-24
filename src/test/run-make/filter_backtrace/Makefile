-include ../tools.mk

# Test that short backtraces don't include several internal symbols

all:
	$(RUSTC) main.rs -o $(TMPDIR)/main
	# with short backtrace
	RUST_BACKTRACE=1 $(TMPDIR)/main 2> $(TMPDIR)/short_bt.stderr || exit 0 && exit 1
	cat $(TMPDIR)/short_bt.stderr # FIXME: remove this debug cat
	$(CGREP) "panicked at" < $(TMPDIR)/short_bt.stderr
	$(CGREP) -v "std::panicking" < $(TMPDIR)/short_bt.stderr
	$(CGREP) -v "std::sys" < $(TMPDIR)/short_bt.stderr
	$(CGREP) -v "__rust_maybe_catch_panic" < $(TMPDIR)/short_bt.stderr
	$(CGREP) -v "__rust_begin_short_backtrace" < $(TMPDIR)/short_bt.stderr
	# with long backtrace
	RUST_BACKTRACE=full $(TMPDIR)/main 2> $(TMPDIR)/long_bt.stderr || exit 0 && exit 1
	cat $(TMPDIR)/long_bt.stderr # FIXME: remove this debug cat
	$(CGREP) "panicked at" < $(TMPDIR)/long_bt.stderr
	$(CGREP) "std::panicking" < $(TMPDIR)/long_bt.stderr
	$(CGREP) "std::sys" < $(TMPDIR)/long_bt.stderr
	$(CGREP) "__rust_maybe_catch_panic" < $(TMPDIR)/long_bt.stderr
	# FIXME: prevent tail call optimization for this
	$(CGREP) -v "__rust_begin_short_backtrace" < $(TMPDIR)/long_bt.stderr
