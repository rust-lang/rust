-include ../tools.mk

all:
	# check that we don't ICE on unicode input, issue #11178
	$(RUSTC) multiple_files.rs
	$(call RUN,multiple_files)  "$(RUSTC)" "$(TMPDIR)"

	# check that our multibyte-ident spans are (approximately) the
	# correct length. issue #8706
	$(RUSTC) span_length.rs
	$(call RUN,span_length) "$(RUSTC)" "$(TMPDIR)"
