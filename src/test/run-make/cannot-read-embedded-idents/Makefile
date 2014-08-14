-include ../tools.mk

# Issue #15750, #15962 : This test ensures that our special embedded
# ident syntax hack is not treated as legitimate input by the lexer in
# normal mode.
#
# It is modelled after the `unicode-input/` test, since we need to
# create files with syntax that can trip up normal text editting tools
# (namely text with embedded nul-bytes).

# This test attempts to run rustc itself from the compiled binary; but
# that means that you need to set the LD_LIBRARY_PATH for rustc itself
# while running create_and_compile, and that won't work for stage1.

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
	$(RUSTC) create_and_compile.rs
	$(call RUN,create_and_compile)  "$(RUSTC)" "$(TMPDIR)"
