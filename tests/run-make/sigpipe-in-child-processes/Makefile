# ignore-cross-compile because we run the compiled code
# only-unix

# See main.rs for a description of this test.

include ../tools.mk

all:
	$(RUSTC) assert-sigpipe-disposition.rs -o $(TMPDIR)/assert-sigpipe-disposition;
	for revision in default sig_dfl sig_ign inherit; do \
		echo -n "Testing revision $$revision ... "; \
		$(RUSTC) main.rs --cfg $$revision -o $(TMPDIR)/main.$$revision || exit 1; \
		$(TMPDIR)/main.$$revision $(TMPDIR)/assert-sigpipe-disposition || exit 1; \
		echo "ok"; \
	done
