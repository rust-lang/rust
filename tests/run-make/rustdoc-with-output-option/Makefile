include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"

all:
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --output $(OUTPUT_DIR)

	$(HTMLDOCCK) $(OUTPUT_DIR) src/lib.rs
