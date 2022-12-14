include ../tools.mk

# only-aarch64

all:
	$(COMPILE_OBJ) $(TMPDIR)/test.o test.c
	$(AR) rcs $(TMPDIR)/libtest.a $(TMPDIR)/test.o
	$(RUSTC) -Z branch-protection=bti,pac-ret,leaf test.rs
	$(call RUN,test)

	$(COMPILE_OBJ) $(TMPDIR)/test.o test.c -mbranch-protection=bti+pac-ret+leaf
	$(AR) rcs $(TMPDIR)/libtest.a $(TMPDIR)/test.o
	$(RUSTC) -Z branch-protection=bti,pac-ret,leaf test.rs
	$(call RUN,test)
