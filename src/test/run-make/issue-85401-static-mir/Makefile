include ../../run-make-fulldeps/tools.mk

# Regression test for issue #85401
# Verify that we do not ICE when trying to access MIR for statics,
# but emit an error when linking.

OUTPUT_FILE := $(TMPDIR)/build-output

all:
	$(RUSTC) --crate-type rlib --crate-name foo -Crelocation-model=pic --edition=2018 foo.rs -Zalways-encode-mir=yes --emit metadata -o $(TMPDIR)/libfoo.rmeta
	$(RUSTC) --crate-type rlib --crate-name bar -Crelocation-model=pic --edition=2018 bar.rs -o $(TMPDIR)/libbar.rlib --extern=foo=$(TMPDIR)/libfoo.rmeta
	$(RUSTC) --crate-type bin --crate-name baz -Crelocation-model=pic --edition=2018 baz.rs -o $(TMPDIR)/baz -L $(TMPDIR) --extern=bar=$(TMPDIR)/libbar.rlib > $(OUTPUT_FILE) 2>&1; [ $$? -eq 1 ]
	cat  $(OUTPUT_FILE)
	$(CGREP) 'crate `foo` required to be available in rlib format, but was not found in this form' < $(OUTPUT_FILE)
	# -v tests are fragile, hopefully this text won't change
	$(CGREP) -v "internal compiler error" < $(OUTPUT_FILE)
