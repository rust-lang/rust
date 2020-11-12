# needs-profiler-support

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../coverage/coverage_tools.mk for more information.

-include ../coverage/coverage_tools.mk

BASEDIR=../coverage-spanview-base
SOURCEDIR=../coverage

ifeq ($(UNAME),Darwin)
SED_HAS_ISSUES=1
endif

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
			-Zdump-mir-spanview \
			-Zdump-mir-dir="$(TMPDIR)"/mir_dump.$@

ifdef SED_HAS_ISSUES
	# FIXME(richkadel): MacOS's default sed has some significant limitations. Until I've come up
	# with a better workaround, I'm disabling this test for MacOS.
	#
	# For future reference, see if `gsed` is available as an alternative.
	which gsed || echo "no gsed"
else

	for path in "$(TMPDIR)"/mir_dump.$@/*; do \
		echo $$path; \
		file="$$(basename "$$path")"; \
		echo $$file; \
		urlescaped="$$("$(PYTHON)" $(BASEDIR)/escape_url.py $$file)" || exit $$?; \
		echo $$urlescaped; \
		sed -i -e '1a\
<!--\
\
Preview this file as rendered HTML from the github source at:\
https://htmlpreview.github.io/?https://github.com/rust-lang/rust/blob/master/src/test/run-make-fulldeps/coverage-spanview-base/expected_mir_dump.$@/'"$$urlescaped"'\
\
For revisions in Pull Requests (PR):\
  * Replace "rust-lang" with the github PR author\
  * Replace "master" with the PR branch name\
\
-->' "$$path"; \
	done && true # for/done ends in non-zero status

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

endif
