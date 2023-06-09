include ../tools.mk

# ignore-cross-compile
# ignore-wasm

all:
	$(RUSTC) --crate-type rlib -lbar_cli bar.rs
	$(RUSTC) foo.rs -lfoo_cli --crate-type staticlib --print native-static-libs 2>&1 \
		| grep 'note: native-static-libs: ' \
		| sed 's/note: native-static-libs: \(.*\)/\1/' > $(TMPDIR)/libs.txt

	cat $(TMPDIR)/libs.txt | grep -F "glib-2.0" # in bar.rs
	cat $(TMPDIR)/libs.txt | grep -F "systemd" # in foo.rs
	cat $(TMPDIR)/libs.txt | grep -F "bar_cli"
	cat $(TMPDIR)/libs.txt | grep -F "foo_cli"
