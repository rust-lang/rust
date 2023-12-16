# only-macos
#
# Check that linking to a framework actually makes it to the linker.

include ../tools.mk

all:
	$(RUSTC) dep-link-framework.rs
	$(RUSTC) dep-link-weak-framework.rs

	$(RUSTC) empty.rs
	otool -L $(TMPDIR)/no-link | $(CGREP) -v CoreFoundation

	$(RUSTC) link-framework.rs
	otool -L $(TMPDIR)/link-framework | $(CGREP) CoreFoundation | $(CGREP) -v weak

	$(RUSTC) link-weak-framework.rs
	otool -L $(TMPDIR)/link-weak-framework | $(CGREP) CoreFoundation | $(CGREP) weak

# When linking the framework both normally, and weakly, the weak linking takes preference

	$(RUSTC) link-both.rs
	otool -L $(TMPDIR)/link-both | $(CGREP) CoreFoundation | $(CGREP) weak
