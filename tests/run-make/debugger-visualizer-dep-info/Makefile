include ../tools.mk

# This test makes sure that files referenced via #[debugger_visualizer] are
# included in `--emit dep-info` output.

all:
	$(RUSTC) --emit dep-info main.rs
	$(CGREP) "foo.py" < $(TMPDIR)/main.d
	$(CGREP) "my_visualizers/bar.natvis" < $(TMPDIR)/main.d
