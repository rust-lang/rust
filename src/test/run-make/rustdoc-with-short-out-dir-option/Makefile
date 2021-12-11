-include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"

all:
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib -o $(OUTPUT_DIR)

	$(HTMLDOCCK) $(OUTPUT_DIR) src/lib.rs
