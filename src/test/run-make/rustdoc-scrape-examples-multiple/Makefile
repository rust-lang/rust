-include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"

all: $(TMPDIR)/ex.calls $(TMPDIR)/ex2.calls
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --output $(OUTPUT_DIR) \
		-Z unstable-options \
		--with-examples $(TMPDIR)/ex.calls \
		--with-examples $(TMPDIR)/ex2.calls

	$(HTMLDOCCK) $(OUTPUT_DIR) src/lib.rs

$(TMPDIR)/%.calls: $(TMPDIR)/libfoobar.rmeta
	$(RUSTDOC) examples/$*.rs --crate-name $* --crate-type bin --output $(OUTPUT_DIR) \
	  --extern foobar=$(TMPDIR)/libfoobar.rmeta \
		-Z unstable-options \
		--scrape-examples-output-path $@ \
		--scrape-examples-target-crate foobar

$(TMPDIR)/lib%.rmeta: src/lib.rs
	$(RUSTC) src/lib.rs --crate-name $* --crate-type lib --emit=metadata
