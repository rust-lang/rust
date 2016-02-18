-include ../tools.mk
all:
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build1"
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build2"
	cmp "$(TMPDIR)/reproducible-build1" "$(TMPDIR)/reproducible-build2" || exit 1
	$(RUSTC) reproducible-build-aux.rs -g
	$(RUSTC) reproducible-build.rs -g -o"$(TMPDIR)/reproducible-build1-debug"
	$(RUSTC) reproducible-build.rs -g -o"$(TMPDIR)/reproducible-build2-debug"
	cmp "$(TMPDIR)/reproducible-build1-debug" "$(TMPDIR)/reproducible-build2-debug" || exit 1
	$(RUSTC) reproducible-build-aux.rs -O
	$(RUSTC) reproducible-build.rs -O -o"$(TMPDIR)/reproducible-build1-opt"
	$(RUSTC) reproducible-build.rs -O -o"$(TMPDIR)/reproducible-build2-opt"
	cmp "$(TMPDIR)/reproducible-build1-opt" "$(TMPDIR)/reproducible-build2-opt" || exit 1
