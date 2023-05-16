# ignore-windows-gnu

include ../tools.mk

all:
	$(RUSTC) --emit dep-info main.rs
	$(CGREP) "foo.py" < $(TMPDIR)/main.d
	$(CGREP) "my_visualizers/bar.py" < $(TMPDIR)/main.d
