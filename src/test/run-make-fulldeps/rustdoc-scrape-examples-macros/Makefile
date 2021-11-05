-include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"
DYLIB_NAME := $(shell echo | $(RUSTC) --crate-name foobar_macro --crate-type dylib --print file-names -)

all:
	$(RUSTC) src/proc.rs --crate-name foobar_macro --edition=2021 --crate-type proc-macro --emit=dep-info,link

	$(RUSTC) src/lib.rs --crate-name foobar --edition=2021 --crate-type lib --emit=dep-info,link

	$(RUSTDOC) examples/ex.rs --crate-name ex --crate-type bin --output $(OUTPUT_DIR) \
		--extern foobar=$(TMPDIR)/libfoobar.rlib --extern foobar_macro=$(TMPDIR)/$(DYLIB_NAME) \
		-Z unstable-options --scrape-examples-output-path $(TMPDIR)/ex.calls --scrape-examples-target-crate foobar

	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --output $(OUTPUT_DIR) \
		-Z unstable-options --with-examples $(TMPDIR)/ex.calls

	$(HTMLDOCCK) $(OUTPUT_DIR) src/lib.rs
