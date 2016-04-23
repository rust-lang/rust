-include ../tools.mk

LOG := $(TMPDIR)/foo.log

all:
	cp foo.rs $(TMPDIR)
	cd $(TMPDIR)
	-$(RUSTC) -Z unstable-options --error-format=json foo.rs 2>$(LOG)
	diff foo.json $(LOG)
