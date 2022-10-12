include ../../run-make-fulldeps/tools.mk

all:
	# Verbatim allows specify precise name.
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/upstream_some_strange_name.ext
	$(RUSTC) rust_dep.rs -Zunstable-options -l static:+verbatim=upstream_some_strange_name.ext --crate-type rlib

	# With verbatim any other name cannot be used (upstream).
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/libupstream_native_dep.a
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/upstream_native_dep.a
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/upstream_native_dep.lib
	$(RUSTC) rust_dep.rs -Zunstable-options -l static:+verbatim=upstream_native_dep --crate-type rlib 2>&1 | $(CGREP) "upstream_native_dep"
