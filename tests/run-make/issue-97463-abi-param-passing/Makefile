# ignore-cross-compile
include ../tools.mk

# ignore-msvc

# The issue exercised by this test, rust-lang/rust#97463, explicitly needs `-O`
# flags (like `-O3`) to reproduce. Thus, we call $(CC) instead of nicer
# alternatives provided by tools.mk like using `COMPILE_OBJ` or using a
# `NATIVE_STATICLIB` dependency.

all:
	$(CC) -c -O3 -o $(TMPDIR)/bad.o bad.c
	$(AR) rcs $(TMPDIR)/libbad.a $(TMPDIR)/bad.o
	$(RUSTC) param_passing.rs -L$(TMPDIR) -lbad -C opt-level=3
	$(call RUN,param_passing)
