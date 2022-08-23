include ../tools.mk

all:
	rm -f $(TMPDIR)/foo-output
	$(RUSTC) -C codegen-units=4 -o $(TMPDIR)/foo-output foo.rs
	rm $(TMPDIR)/foo-output

	rm -f $(TMPDIR)/asm-output
	$(RUSTC) -C codegen-units=4 --emit=asm -o $(TMPDIR)/asm-output foo.rs
	rm $(TMPDIR)/asm-output

	rm -f $(TMPDIR)/bc-output
	$(RUSTC) -C codegen-units=4 --emit=llvm-bc -o $(TMPDIR)/bc-output foo.rs
	rm $(TMPDIR)/bc-output

	rm -f $(TMPDIR)/ir-output
	$(RUSTC) -C codegen-units=4 --emit=llvm-ir -o $(TMPDIR)/ir-output foo.rs
	rm $(TMPDIR)/ir-output

	rm -f $(TMPDIR)/link-output
	$(RUSTC) -C codegen-units=4 --emit=link -o $(TMPDIR)/link-output foo.rs
	rm $(TMPDIR)/link-output

	rm -f $(TMPDIR)/obj-output
	$(RUSTC) -C codegen-units=4 --emit=obj -o $(TMPDIR)/obj-output foo.rs
	rm $(TMPDIR)/obj-output

	rm -f $(TMPDIR)/dep-output
	$(RUSTC) -C codegen-units=4 --emit=dep-info -o $(TMPDIR)/dep-output foo.rs
	rm $(TMPDIR)/dep-output

#	# (This case doesn't work yet, and may be fundamentally wrong-headed anyway.)
#	rm -f $(TMPDIR)/multi-output
#	$(RUSTC) -C codegen-units=4 --emit=asm,obj -o $(TMPDIR)/multi-output foo.rs
#	rm $(TMPDIR)/multi-output
