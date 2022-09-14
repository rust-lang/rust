# ignore-cross-compile
# ignore-macos

include ../../run-make-fulldeps/tools.mk

all:
	# Verbatim allows specify precise name.
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/local_some_strange_name.ext
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/upstream_some_strange_name.ext
	$(RUSTC) rust_dep.rs -Zunstable-options -l static:+verbatim=upstream_some_strange_name.ext --crate-type rlib
	$(RUSTC) main.rs --extern rust_dep=$(TMPDIR)/librust_dep.rlib -Zunstable-options -l static:+verbatim=local_some_strange_name.ext
	rm $(TMPDIR)/*some_strange_name.ext

	# With verbatim any other name cannot be used (local).
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/liblocal_native_dep.a
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/local_native_dep.a
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/upstream_native_dep
	$(RUSTC) rust_dep.rs -Zunstable-options -l static:+verbatim=upstream_native_dep --crate-type rlib
	$(RUSTC) main.rs --extern rust_dep=$(TMPDIR)/librust_dep.rlib -Zunstable-options -l static:+verbatim=local_native_dep 2>&1 | $(CGREP) "local_native_dep"
	rm $(TMPDIR)/*native_dep*

	# With verbatim any other name cannot be used (upstream).
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/local_native_dep
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/libupstream_native_dep.a
	$(RUSTC) upstream_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/upstream_native_dep.a
	$(RUSTC) rust_dep.rs -Zunstable-options -l static:+verbatim=upstream_native_dep --crate-type rlib 2>&1 | $(CGREP) "upstream_native_dep"
	rm $(TMPDIR)/*native_dep*
