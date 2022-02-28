-include ../../run-make-fulldeps/tools.mk

# Tests that the `-Z force-allocator-shim` flag causes an allocator shim to be
# generated when expected.

all:
	$(RUSTC) foo.rs --crate-type rlib -Z force-allocator-shim
	nm $(TMPDIR)/libfoo.rlib | $(CGREP) "__rust_alloc" && \
	nm $(TMPDIR)/libfoo.rlib | $(CGREP) "__rust_alloc_error_handler" && \
	nm $(TMPDIR)/libfoo.rlib | $(CGREP) "__rust_alloc_zeroed" && \
	nm $(TMPDIR)/libfoo.rlib | $(CGREP) "__rust_dealloc" && \
	nm $(TMPDIR)/libfoo.rlib | $(CGREP) "__rust_realloc"
