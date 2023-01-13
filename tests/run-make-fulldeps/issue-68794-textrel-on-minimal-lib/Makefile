# Regression test for issue #68794
#
# Verify that no text relocations are accidentally introduced by linking a
# minimal rust staticlib.
#
# The test links a rust static library into a shared library, and checks that
# the linker doesn't have to flag the resulting file as containing TEXTRELs.

include ../tools.mk

# only-linux

all:
	$(RUSTC) foo.rs
	$(CC) bar.c $(call STATICLIB,foo) -fPIC -shared -o $(call DYLIB,bar) \
		$(EXTRACFLAGS) $(EXTRACXXFLAGS)
	readelf -d $(call DYLIB,bar) | grep TEXTREL; test $$? -eq 1
