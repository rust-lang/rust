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

	# TODO
	# We should re-generate json doc once again and compare the diff with previously
	# generated one. Because layout of json docs changes in each compilation, we can't
	# do that currently.
	#
	# See https://github.com/rust-lang/rust/issues/103785#issuecomment-1307425590 for details.

	# remove generated json doc
	rm $(OUTPUT_DIR)/foobar.json

	# Check if json doc compilation broke any of the html files generated previously
	$(DIFF) -r -q $(OUTPUT_DIR) $(TMP_OUTPUT_DIR)
