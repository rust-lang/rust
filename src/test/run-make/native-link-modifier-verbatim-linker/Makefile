# ignore-cross-compile
# ignore-macos

include ../../run-make-fulldeps/tools.mk

all:
	# Verbatim allows specify precise name.
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/local_some_strange_name.ext
	$(RUSTC) main.rs -Zunstable-options -l static:+verbatim=local_some_strange_name.ext

	# With verbatim any other name cannot be used (local).
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/liblocal_native_dep.a
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/local_native_dep.a
	$(RUSTC) local_native_dep.rs --crate-type=staticlib -o $(TMPDIR)/local_native_dep.lib
	$(RUSTC) main.rs -Zunstable-options -l static:+verbatim=local_native_dep 2>&1 | $(CGREP) "local_native_dep"
