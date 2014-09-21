-include ../tools.mk

# This test attempts to use syntax extensions, which are known to be
# incompatible with stage1 at the moment.

ifeq ($(RUST_BUILD_STAGE),1)
DOTEST=
else
DOTEST=dotest
endif

all: $(DOTEST)

dotest:
	env
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto
	$(call RUN,main)
