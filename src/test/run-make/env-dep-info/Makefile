-include ../../run-make-fulldeps/tools.mk

all:
	EXISTING_ENV=1 EXISTING_OPT_ENV=1 $(RUSTC) --emit dep-info main.rs
	$(CGREP) "# env-dep:EXISTING_ENV=1" < $(TMPDIR)/main.d
	$(CGREP) "# env-dep:EXISTING_OPT_ENV=1" < $(TMPDIR)/main.d
	$(CGREP) "# env-dep:NONEXISTENT_OPT_ENV" < $(TMPDIR)/main.d
	$(CGREP) "# env-dep:ESCAPE\nESCAPE\\" < $(TMPDIR)/main.d
