include ../tools.mk

all:
	# Work in /tmp, because we need to create the `save-analysis-temp` folder.
	cp a.rs $(TMPDIR)/
	cd $(TMPDIR) && $(RUSTC) -Zsave-analysis $(TMPDIR)/a.rs 2> $(TMPDIR)/stderr.txt || ( cat $(TMPDIR)/stderr.txt && exit 1 )
	[ ! -s $(TMPDIR)/stderr.txt ] || ( cat $(TMPDIR)/stderr.txt && exit 1 )
	[ -f $(TMPDIR)/save-analysis/liba.json ] || ( ls -la $(TMPDIR) && exit 1 )
