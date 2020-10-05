# needs-profiler-support

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../coverage/coverage_tools.mk for more information.

-include ../coverage/coverage_tools.mk

SOURCEDIR=../coverage

all: $(patsubst $(SOURCEDIR)/%.rs,%,$(wildcard $(SOURCEDIR)/*.rs))

# Ensure there are no `expected` results for tests that may have been removed or renamed
.PHONY: clear_expected_if_blessed
clear_expected_if_blessed:
ifdef RUSTC_BLESS_TEST
	rm -rf expected_mir_dump.*/
endif

-include clear_expected_if_blessed

%: $(SOURCEDIR)/%.rs
	# Compile the test program with coverage instrumentation and generate relevant MIR.
	$(RUSTC) $(SOURCEDIR)/$@.rs \
			-Zinstrument-coverage \
			-Clink-dead-code=$(LINK_DEAD_CODE) \
			-Zdump-mir=InstrumentCoverage \
			-Zdump-mir-dir="$(TMPDIR)"/mir_dump.$@

ifdef RUSTC_BLESS_TEST
	mkdir -p expected_mir_dump.$@
	cp "$(TMPDIR)"/mir_dump.$@/*InstrumentCoverage.0.html expected_mir_dump.$@/
else
	# Check that the selected `mir_dump` files match what we expect (`--bless` refreshes `expected`)
	mkdir -p "$(TMPDIR)"/actual_mir_dump.$@
	rm -f "$(TMPDIR)"/actual_mir_dump.$@/*
	cp "$(TMPDIR)"/mir_dump.$@/*InstrumentCoverage.0.html "$(TMPDIR)"/actual_mir_dump.$@/
	$(DIFF) -r expected_mir_dump.$@/ "$(TMPDIR)"/actual_mir_dump.$@/
endif
