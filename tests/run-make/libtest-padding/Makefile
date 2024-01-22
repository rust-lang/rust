# ignore-cross-compile because we run the compiled code
# needs-unwind because #[bench] and -Cpanic=abort requires -Zpanic-abort-tests
include ../tools.mk

NORMALIZE=sed 's%[0-9,]\{1,\} ns/iter (+/- [0-9,]\{1,\})%?? ns/iter (+/- ??)%' | sed 's%finished in [0-9\.]\{1,\}%finished in ??%'

all:
	$(RUSTC) --test tests.rs

	$(call RUN,tests) --test-threads=1 | $(NORMALIZE) > "$(TMPDIR)"/test.stdout
	$(RUSTC_TEST_OP) "$(TMPDIR)"/test.stdout test.stdout

	$(call RUN,tests) --test-threads=1 --bench | $(NORMALIZE) > "$(TMPDIR)"/bench.stdout
	$(RUSTC_TEST_OP) "$(TMPDIR)"/bench.stdout bench.stdout
