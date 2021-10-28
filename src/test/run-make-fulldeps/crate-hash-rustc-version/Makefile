-include ../../run-make-fulldeps/tools.mk

# Ensure that crates compiled with different rustc versions cannot
# be dynamically linked.

FLAGS := -Cprefer-dynamic -Zsymbol-mangling-version=v0
UNAME := $(shell uname)
ifeq ($(UNAME),Linux)
  EXT=".so"
  NM_CMD := nm -D
endif
ifeq ($(UNAME),Darwin)
  EXT=".dylib"
  NM_CMD := nm
endif

ifndef NM_CMD
all:
	exit 0
else
all:
	# a.rs is a dylib
	$(RUSTC) a.rs --crate-type=dylib $(FLAGS)
	# Write symbols to disk.
	$(NM_CMD) $(call DYLIB,a) > $(TMPDIR)/symbolsbefore
	# b.rs is a binary
	$(RUSTC) b.rs --extern a=$(TMPDIR)/liba$(EXT) --crate-type=bin -Crpath $(FLAGS)
	$(call RUN,b)
	# Now re-compile a.rs with another rustc version
	RUSTC_FORCE_INCR_COMP_ARTIFACT_HEADER=deadfeed $(RUSTC) a.rs --crate-type=dylib $(FLAGS)
	# After compiling with a different rustc version, write symbols to disk again.
	$(NM_CMD) $(call DYLIB,a) > $(TMPDIR)/symbolsafter
	# As a sanity check, test if the symbols changed:
	# If the symbols are identical, there's been an error.
	if diff $(TMPDIR)/symbolsbefore $(TMPDIR)/symbolsafter; then exit 1; fi
	$(call FAIL,b)
endif
