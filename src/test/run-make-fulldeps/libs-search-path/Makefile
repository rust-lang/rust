-include ../tools.mk

# only-mingw

all: empty.rs
	cp -r $(shell cygpath -u $(shell $(RUSTC) --print sysroot)) $(TMPDIR)/sysroot
	$(RUSTC) --target $(TARGET) --sysroot $(TMPDIR)/sysroot -L$(TMPDIR)/obj -Z print-link-args empty.rs | $(CGREP) 'lib\\crt2.o'
	mkdir -p $(TMPDIR)/obj
	mv $(TMPDIR)/sysroot/lib/rustlib/$(TARGET)/lib/crt2.o $(TMPDIR)/obj/crt2.o
	$(RUSTC) --target $(TARGET) --sysroot $(TMPDIR)/sysroot -L$(TMPDIR)/obj -Z print-link-args empty.rs | $(CGREP) 'obj\\crt2.o'
