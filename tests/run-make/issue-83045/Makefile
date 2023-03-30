include ../../run-make-fulldeps/tools.mk

# This test case creates a situation where the crate loader would run
# into an ICE when confronted with an invalid setup where it cannot
# find the dependency of a direct dependency.
#
# The test case makes sure that the compiler produces the expected
# error message but does not ICE immediately after.
#
# See https://github.com/rust-lang/rust/issues/83045

# This is a platform-independent issue, no need to waste time testing
# everywhere.
# only-x86_64
# only-linux

# NOTE: We use BARE_RUSTC below so that the compiler can't find liba.rlib
#       If we used RUSTC the additional '-L TMPDIR' option would allow rustc to
#       actually find the crate.
#
#       We check that we get the expected error message
#       But that we do not get an ICE

all:
	$(RUSTC) --crate-name=a --crate-type=rlib a.rs --verbose
	$(RUSTC) --crate-name=b --crate-type=rlib --extern a=$(TMPDIR)/liba.rlib b.rs --verbose
	$(BARE_RUSTC) --out-dir $(TMPDIR) \
	              --extern b=$(TMPDIR)/libb.rlib \
				  --crate-type=rlib \
				  --edition=2018 \
				  c.rs 2>&1 | tee $(TMPDIR)/output.txt || exit 0
	$(CGREP) E0519 < $(TMPDIR)/output.txt
	$(CGREP) -v "internal compiler error" < $(TMPDIR)/output.txt
