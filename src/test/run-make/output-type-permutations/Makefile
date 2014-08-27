-include ../tools.mk

all:
	$(RUSTC) foo.rs --crate-type=rlib,dylib,staticlib
	$(call REMOVE_RLIBS,bar)
	$(call REMOVE_DYLIBS,bar)
	rm $(TMPDIR)/$(call STATICLIB_GLOB,bar)
	# Check that $(TMPDIR) is empty.
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=bin
	rm $(TMPDIR)/$(call BIN,bar)
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=asm,ir,bc,obj,link
	rm $(TMPDIR)/bar.ll
	rm $(TMPDIR)/bar.bc
	rm $(TMPDIR)/bar.s
	rm $(TMPDIR)/bar.o
	rm $(TMPDIR)/$(call BIN,bar)
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=asm -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=bc -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=ir -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=obj -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=link -o $(TMPDIR)/foo
	rm $(TMPDIR)/$(call BIN,foo)
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=rlib -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=dylib -o $(TMPDIR)/foo
	rm $(TMPDIR)/$(call BIN,foo)  # FIXME 13794
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=staticlib -o $(TMPDIR)/foo
	rm $(TMPDIR)/foo
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --crate-type=bin -o $(TMPDIR)/foo
	rm $(TMPDIR)/$(call BIN,foo)
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]

	$(RUSTC) foo.rs --emit=asm,ir,bc,obj,link --crate-type=staticlib
	rm $(TMPDIR)/bar.ll
	rm $(TMPDIR)/bar.s
	rm $(TMPDIR)/bar.o
	rm $(TMPDIR)/$(call STATICLIB_GLOB,bar)
	mv $(TMPDIR)/bar.bc $(TMPDIR)/foo.bc
	# Don't check that the $(TMPDIR) is empty - we left `foo.bc` for later
	# comparison.

	$(RUSTC) foo.rs --emit=bc,link --crate-type=rlib
	cmp $(TMPDIR)/foo.bc $(TMPDIR)/bar.bc
	rm $(TMPDIR)/bar.bc
	rm $(TMPDIR)/foo.bc
	$(call REMOVE_RLIBS,bar)
	[ "$$(ls -1 $(TMPDIR) | wc -l)" -eq "0" ]
