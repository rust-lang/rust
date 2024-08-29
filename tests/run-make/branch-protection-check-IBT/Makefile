# Check for GNU Property Note

include ../tools.mk

# How to run this
# python3 x.py test --target x86_64-unknown-linux-gnu  tests/run-make/branch-protection-check-IBT/

# only-x86_64

# ignore-test
# FIXME(jieyouxu): This test never runs because the `ifeq` check on line 17
# compares `x86` to `x86_64`, which always evaluates to false.
# When the test does run, the compilation does not include `.note.gnu.property`.
# See https://github.com/rust-lang/rust/pull/126720 for more information.

all:
ifeq ($(filter x86,$(LLVM_COMPONENTS)),x86_64)
	$(RUSTC) --target x86_64-unknown-linux-gnu -Z cf-protection=branch -L$(TMPDIR) -C link-args='-nostartfiles'  -C save-temps  ./main.rs -o $(TMPDIR)/rsmain
	 readelf -nW $(TMPDIR)/rsmain | $(CGREP) -e ".note.gnu.property"
endif

