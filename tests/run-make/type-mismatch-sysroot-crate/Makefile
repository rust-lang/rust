# Test that we emit useful diagnostics when the same crate is loaded from the sysroot and --extern.
# This can't be a run-make test because it needs the aux-file to itself have a dependency passed with --extern
# (and just bare `--extern hashbrown` errors too early, while compiling `uses_hashbrown.rs`).

include ../tools.mk
define SED_REGEX
s#/hashbrown-[^/]*/src#/hashbrown-VERSION/src#;
s#[^ ]*/build/[^ ]*/lib/rustlib/[^ ]*#BUILD_DIR/HOST/stageN/lib/rustlib/TARGET/lib/libhashbrown.rlib#;
s#[^ ]*/build/[^ ]*/test/run-make/#BUILD_DIR/HOST/test/run-make/#
s#[^ ]*tests/run-make/#TEST_DIR/#
endef
export SED_REGEX

all:
	$(RUSTC) hashbrown.rs --crate-type lib
	$(RUSTC) uses-hashbrown.rs --extern hashbrown=$(TMPDIR)/libhashbrown.rlib --crate-type lib
	$(RUSTC) mismatch.rs --extern uses_hashbrown=$(TMPDIR)/libuses_hashbrown.rlib 2>$(TMPDIR)/stderr.txt || true
	sed -e "$$SED_REGEX" < $(TMPDIR)/stderr.txt > $(TMPDIR)/normalized.txt
	$(RUSTC_TEST_OP) $(TMPDIR)/normalized.txt expected.txt
	$(CGREP) "loaded from" < $(TMPDIR)/stderr.txt
