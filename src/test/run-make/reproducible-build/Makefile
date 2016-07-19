-include ../tools.mk
all:
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build1"
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build2"
	nm "$(TMPDIR)/reproducible-build1" | sort > "$(TMPDIR)/reproducible-build1.nm"
	nm "$(TMPDIR)/reproducible-build2" | sort > "$(TMPDIR)/reproducible-build2.nm"
	cmp "$(TMPDIR)/reproducible-build1.nm" "$(TMPDIR)/reproducible-build2.nm" || exit 1
	$(RUSTC) reproducible-build-aux.rs -g
	$(RUSTC) reproducible-build.rs -g -o"$(TMPDIR)/reproducible-build1-debug"
	$(RUSTC) reproducible-build.rs -g -o"$(TMPDIR)/reproducible-build2-debug"
	nm "$(TMPDIR)/reproducible-build1-debug" | sort > "$(TMPDIR)/reproducible-build1-debug.nm"
	nm "$(TMPDIR)/reproducible-build2-debug" | sort > "$(TMPDIR)/reproducible-build2-debug.nm"
	cmp "$(TMPDIR)/reproducible-build1-debug.nm" "$(TMPDIR)/reproducible-build2-debug.nm" || exit 1
	$(RUSTC) reproducible-build-aux.rs -O
	$(RUSTC) reproducible-build.rs -O -o"$(TMPDIR)/reproducible-build1-opt"
	$(RUSTC) reproducible-build.rs -O -o"$(TMPDIR)/reproducible-build2-opt"
	nm "$(TMPDIR)/reproducible-build1-opt" | sort > "$(TMPDIR)/reproducible-build1-opt.nm"
	nm "$(TMPDIR)/reproducible-build2-opt" | sort > "$(TMPDIR)/reproducible-build2-opt.nm"
	cmp "$(TMPDIR)/reproducible-build1-opt.nm" "$(TMPDIR)/reproducible-build2-opt.nm" || exit 1
