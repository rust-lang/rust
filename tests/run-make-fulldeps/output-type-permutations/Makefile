include ../tools.mk

all:
	$(RUSTC) foo.rs --crate-type=rlib,dylib,staticlib
	$(call REMOVE_RLIBS,bar)
	$(call REMOVE_DYLIBS,bar)
	rm $(call STATICLIB,bar)
	rm -f $(TMPDIR)/{lib,}bar.{dll.exp,dll.lib,pdb,dll.a}
	# Check that $(TMPDIR) is empty.
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=bin
	rm $(TMPDIR)/$(call BIN,bar)
	rm -f $(TMPDIR)/bar.pdb
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=asm,llvm-ir,llvm-bc,obj,link
	rm $(TMPDIR)/bar.ll
	rm $(TMPDIR)/bar.bc
	rm $(TMPDIR)/bar.s
	rm $(TMPDIR)/bar.o
	rm $(TMPDIR)/$(call BIN,bar)
	rm -f $(TMPDIR)/bar.pdb
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit asm -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit asm=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=asm=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit llvm-bc -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit llvm-bc=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=llvm-bc=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit llvm-ir -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit llvm-ir=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=llvm-ir=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit obj -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit obj=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --emit=obj=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit link -o $(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --emit link=$(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --emit=link=$(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	rm -f $(TMPDIR)/foo.pdb
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=rlib -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --crate-type=rlib --emit link=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --crate-type=rlib --emit=link=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=dylib -o $(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --crate-type=dylib --emit link=$(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --crate-type=dylib --emit=link=$(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	rm -f $(TMPDIR)/{lib,}foo.{dll.exp,dll.lib,pdb,dll.a,exe.a}
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ] || (ls -1 $(TMPDIR) && exit 1)

	$(RUSTC) foo.rs --crate-type=staticlib -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --crate-type=staticlib --emit link=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	$(RUSTC) foo.rs --crate-type=staticlib --emit=link=$(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=bin -o $(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --crate-type=bin --emit link=$(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --crate-type=bin --emit=link=$(TMPDIR)/$(call BIN,foo)
	rm $(TMPDIR)/$(call BIN,foo)
	rm -f $(TMPDIR)/foo.pdb
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit llvm-ir=$(TMPDIR)/ir \
			--emit link \
			--crate-type=rlib
	rm $(TMPDIR)/ir
	rm $(TMPDIR)/libbar.rlib
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit asm=$(TMPDIR)/asm \
			--emit llvm-ir=$(TMPDIR)/ir \
			--emit llvm-bc=$(TMPDIR)/bc \
		        --emit obj=$(TMPDIR)/obj \
			--emit link=$(TMPDIR)/link \
			--crate-type=staticlib
	rm $(TMPDIR)/asm
	rm $(TMPDIR)/ir
	rm $(TMPDIR)/bc
	rm $(TMPDIR)/obj
	rm $(TMPDIR)/link
	$(RUSTC) foo.rs --emit=asm=$(TMPDIR)/asm \
			--emit llvm-ir=$(TMPDIR)/ir \
			--emit=llvm-bc=$(TMPDIR)/bc \
		        --emit obj=$(TMPDIR)/obj \
			--emit=link=$(TMPDIR)/link \
			--crate-type=staticlib
	rm $(TMPDIR)/asm
	rm $(TMPDIR)/ir
	rm $(TMPDIR)/bc
	rm $(TMPDIR)/obj
	rm $(TMPDIR)/link
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=asm,llvm-ir,llvm-bc,obj,link --crate-type=staticlib
	rm $(TMPDIR)/bar.ll
	rm $(TMPDIR)/bar.s
	rm $(TMPDIR)/bar.o
	rm $(call STATICLIB,bar)
	mv $(TMPDIR)/bar.bc $(TMPDIR)/foo.bc
	# Don't check that the $(TMPDIR) is empty - we left `foo.bc` for later
	# comparison.

	$(RUSTC) foo.rs --emit=llvm-bc,link --crate-type=rlib
	cmp $(TMPDIR)/foo.bc $(TMPDIR)/bar.bc
	rm $(TMPDIR)/bar.bc
	rm $(TMPDIR)/foo.bc
	$(call REMOVE_RLIBS,bar)
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]
