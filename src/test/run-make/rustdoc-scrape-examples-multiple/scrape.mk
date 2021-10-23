-include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"

$(TMPDIR)/%.calls: $(TMPDIR)/libfoobar.rmeta
	$(RUSTDOC) examples/$*.rs --crate-name $* --crate-type bin --output $(OUTPUT_DIR) \
	  --extern foobar=$(TMPDIR)/libfoobar.rmeta \
		-Z unstable-options \
		--scrape-examples-output-path $@ \
		--scrape-examples-target-crate foobar

$(TMPDIR)/lib%.rmeta: src/lib.rs
	$(RUSTC) src/lib.rs --crate-name $* --crate-type lib --emit=metadata

scrape: $(foreach d,$(deps),$(TMPDIR)/$(d).calls)
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --output $(OUTPUT_DIR) \
		-Z unstable-options \
		$(foreach d,$(deps),--with-examples $(TMPDIR)/$(d).calls)

	$(HTMLDOCCK) $(OUTPUT_DIR) src/lib.rs
