-include ../tools.mk

# check that rustc builds all crate_type attributes
# delete rlib
# delete whatever dylib is made for this system
# check that rustc only builds --crate-type flags, ignoring attributes
# fail if an rlib was built
all:
	$(RUSTC) test.rs
	rm $(TMPDIR)/$(call RLIB_GLOB,test)
	rm $(TMPDIR)/$(call DYLIB_GLOB,test)
	$(RUSTC) --crate-type dylib test.rs
	rm $(TMPDIR)/$(call RLIB_GLOB,test) && exit 1 || exit 0
