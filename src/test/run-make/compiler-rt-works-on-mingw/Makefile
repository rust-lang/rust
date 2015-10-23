-include ../tools.mk

ifneq (,$(findstring MINGW,$(UNAME)))
ifndef IS_MSVC
all:
	g++ foo.cpp -c -o $(TMPDIR)/foo.o
	ar crus $(TMPDIR)/libfoo.a $(TMPDIR)/foo.o
	$(RUSTC) foo.rs -lfoo -lstdc++
	$(call RUN,foo)
else
all:

endif
else
all:

endif
