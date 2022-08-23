include ../tools.mk

all:
	cp foo.rs $(TMPDIR)
	cd $(TMPDIR) && $(RUSTC) -o foo foo.rs
	$(call RUN,foo)
