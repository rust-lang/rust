-include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"

all:
# 1. compile the library crate and emit an rmeta
	$(RUSTC) src/lib.rs --crate-name foobar --crate-type lib --emit=metadata

# 2. scrape examples from the reverse-dependency into an ex.calls file
	$(RUSTDOC) examples/ex.rs --crate-name ex --crate-type bin --output $(OUTPUT_DIR) \
	  --extern foobar=$(TMPDIR)/libfoobar.rmeta \
		-Z unstable-options \
		--scrape-examples-output-path $(TMPDIR)/ex.calls \
		--scrape-examples-target-crate foobar

# 3. pass those examples to rustdoc when documenting the library crate
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --output $(OUTPUT_DIR) \
		-Z unstable-options \
		--with-examples $(TMPDIR)/ex.calls

# 4. check that the examples were scraped successfully
	$(HTMLDOCCK) $(OUTPUT_DIR) src/lib.rs
