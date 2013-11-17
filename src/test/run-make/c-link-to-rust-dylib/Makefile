-include ../tools.mk

all:
	$(RUSTC) foo.rs
	ln -s $(call DYLIB,foo-*) $(call DYLIB,foo)
	$(CC) bar.c -lfoo -o $(call RUN,bar) -Wl,-rpath,$(TMPDIR)
	$(call RUN,bar)
	rm $(call DYLIB,foo)
	$(call FAIL,bar)
