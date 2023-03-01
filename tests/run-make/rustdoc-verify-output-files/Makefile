include ../../run-make-fulldeps/tools.mk

OUTPUT_DIR := "$(TMPDIR)/rustdoc"
TMP_OUTPUT_DIR := "$(TMPDIR)/tmp-rustdoc"

all:
	# Generate html docs
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --out-dir $(OUTPUT_DIR)

	# Copy first output for to check if it's exactly same after second compilation
	cp -R $(OUTPUT_DIR) $(TMP_OUTPUT_DIR)

	# Generate html docs once again on same output
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --out-dir $(OUTPUT_DIR)

	# Check if everything exactly same
	$(DIFF) -r -q $(OUTPUT_DIR) $(TMP_OUTPUT_DIR)

	# Generate json doc on the same output
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --out-dir $(OUTPUT_DIR) -Z unstable-options --output-format json

	# Check if expected json file is generated
	[ -e $(OUTPUT_DIR)/foobar.json ]

	# Copy first json output to check if it's exactly same after second compilation
	cp -R $(OUTPUT_DIR)/foobar.json $(TMP_OUTPUT_DIR)/foobar.json

	# Generate json doc on the same output
	$(RUSTDOC) src/lib.rs --crate-name foobar --crate-type lib --out-dir $(OUTPUT_DIR) -Z unstable-options --output-format json

	# Check if all docs(including both json and html formats) are still the same after multiple compilations
	$(DIFF) -r -q $(OUTPUT_DIR) $(TMP_OUTPUT_DIR)
