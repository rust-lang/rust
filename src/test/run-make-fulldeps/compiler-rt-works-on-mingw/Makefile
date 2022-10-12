include ../tools.mk

# only-windows-gnu

all:
	$(CXX) foo.cpp -c -o $(TMPDIR)/foo.o
	$(AR) crus $(TMPDIR)/libfoo.a $(TMPDIR)/foo.o
	$(RUSTC) foo.rs -lfoo -lstdc++
	$(call RUN,foo)
