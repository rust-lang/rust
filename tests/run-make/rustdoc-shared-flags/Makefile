include ../tools.mk

all: z_help c_help list_passes

c_help:
	$(RUSTC) -C help > $(TMPDIR)/rustc.c_help.txt
	$(RUSTDOC) -C help > $(TMPDIR)/rustdoc.c_help.txt
	$(DIFF) $(TMPDIR)/rustc.c_help.txt $(TMPDIR)/rustdoc.c_help.txt

z_help:
	$(RUSTC) -Z help > $(TMPDIR)/rustc.z_help.txt
	$(RUSTDOC) -Z help > $(TMPDIR)/rustdoc.z_help.txt
	$(DIFF) $(TMPDIR)/rustc.z_help.txt $(TMPDIR)/rustdoc.z_help.txt

list_passes:
	$(RUSTC) -C passes=list > $(TMPDIR)/rustc.passes.txt
	$(RUSTDOC) -C passes=list > $(TMPDIR)/rustdoc.passes.txt
	$(DIFF) $(TMPDIR)/rustc.passes.txt $(TMPDIR)/rustdoc.passes.txt
