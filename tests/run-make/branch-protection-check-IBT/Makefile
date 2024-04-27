# Check for GNU Property Note

include ../tools.mk

# How to run this
# python3 x.py test --target x86_64-unknown-linux-gnu  tests/run-make/branch-protection-check-IBT/

# only-x86_64

all:
ifeq ($(filter x86,$(LLVM_COMPONENTS)),x86_64)
	$(RUSTC) --target x86_64-unknown-linux-gnu -Z cf-protection=branch -L$(TMPDIR) -C link-args='-nostartfiles'  -C save-temps  ./main.rs -o $(TMPDIR)/rsmain
	 readelf -nW $(TMPDIR)/rsmain | $(CGREP) -e ".note.gnu.property"
endif

