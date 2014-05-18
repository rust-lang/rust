-include ../tools.mk

# This test attempts to run rustc itself from the compiled binary; but
# that means that you need to set the LD_LIBRARY_PATH for rustc itself
# while running multiple_files, and that won't work for stage1.

# FIXME ignore windows
ifndef IS_WINDOWS
ifeq ($(RUST_BUILD_STAGE),1)
DOTEST=
else
DOTEST=dotest
endif
endif

all: $(DOTEST)

dotest:
	# check that we don't ICE on unicode input, issue #11178
	$(RUSTC) multiple_files.rs
	$(call RUN,multiple_files)  "$(RUSTC)" "$(TMPDIR)"

	# check that our multibyte-ident spans are (approximately) the
	# correct length. issue #8706
	$(RUSTC) span_length.rs
	$(call RUN,span_length) "$(RUSTC)" "$(TMPDIR)"
