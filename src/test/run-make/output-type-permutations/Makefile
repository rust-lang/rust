-include ../tools.mk

all:
	$(RUSTC) foo.rs --crate-type=rlib,dylib,staticlib
	rm $(TMPDIR)/$(call RLIB_GLOB,bar)
	rm $(TMPDIR)/$(call DYLIB_GLOB,bar)
	rm $(TMPDIR)/$(call STATICLIB_GLOB,bar)
	$(RUSTC) foo.rs --crate-type=bin
	rm $(TMPDIR)/$(call BIN,bar)
	$(RUSTC) foo.rs --emit=asm,ir,bc,obj,link
	rm $(TMPDIR)/bar.ll
	rm $(TMPDIR)/bar.bc
	rm $(TMPDIR)/bar.s
	rm $(TMPDIR)/bar.o
	rm $(TMPDIR)/$(call BIN,bar)
	$(RUSTC) foo.rs --emit=asm,ir,bc,obj,link --crate-type=staticlib
	rm $(TMPDIR)/bar.ll
	rm $(TMPDIR)/bar.s
	rm $(TMPDIR)/bar.o
	rm $(TMPDIR)/$(call STATICLIB_GLOB,bar)
	$(RUSTC) foo.rs --emit=asm -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=bc -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=ir -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=obj -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=link -o $(TMPDIR)/foo
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --crate-type=rlib -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --crate-type=dylib -o $(TMPDIR)/foo
	rm $(TMPDIR)/$(call BIN,foo)  # FIXME 13794
	$(RUSTC) foo.rs --crate-type=staticlib -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --crate-type=bin -o $(TMPDIR)/foo
	rm $(TMPDIR)/$(call BIN,foo)
	mv $(TMPDIR)/bar.bc $(TMPDIR)/foo.bc
	$(RUSTC) foo.rs --emit=bc,link --crate-type=rlib
	cmp $(TMPDIR)/foo.bc $(TMPDIR)/bar.bc
	rm $(TMPDIR)/bar.bc
	rm $(TMPDIR)/foo.bc
	rm $(TMPDIR)/$(call RLIB_GLOB,bar)
