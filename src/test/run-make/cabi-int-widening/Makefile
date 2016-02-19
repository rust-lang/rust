-include ../tools.mk

all: $(call NATIVE_STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(call RUN,foo)

# Need to optimize the C code so it actually takes advantage
# of the ABI semantics
$(TMPDIR)/libcfoo.o: cfoo.c
	$(call COMPILE_OBJ,$(TMPDIR)/libcfoo.o,cfoo.c -O)
