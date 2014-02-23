-include ../tools.mk

all:
	# check that we don't ICE on unicode input, issue #11178
	$(RUSTC) multiple_files.rs
	$(call RUN,multiple_files)  "$(RUSTC)" "$(TMPDIR)"
