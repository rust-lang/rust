# ignore-cross-compile
include ../tools.mk

# Test that using rlibs and rmeta dep crates work together. Specifically, that
# there can be both an rmeta and an rlib file and rustc will prefer the rmeta
# file.
#
# This behavior is simply making sure this doesn't accidentally change; in this
# case we want to make sure that the rlib isn't being used as that would cause
# bugs in -Zbinary-dep-depinfo (see #68298).

all:
	$(RUSTC) rmeta_aux.rs --crate-type=rlib --emit link,metadata
	$(RUSTC) lib.rs --crate-type=rlib --emit dep-info -Zbinary-dep-depinfo
	$(CGREP) "librmeta_aux.rmeta" < $(TMPDIR)/lib.d
	$(CGREP) -v "librmeta_aux.rlib" < $(TMPDIR)/lib.d
