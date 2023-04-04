include ../../run-make/tools.mk

# ignore-stage1

# This test both exists as a check that -Zcodegen-backend is capable of loading external codegen
# backends and that this external codegen backend is only included in the dep info if
# -Zbinary-dep-depinfo is used.

all:
	/bin/echo || exit 0 # This test requires /bin/echo to exist
	$(RUSTC) the_backend.rs --crate-name the_backend --crate-type dylib \
		-o $(TMPDIR)/the_backend.dylib

	$(RUSTC) some_crate.rs --crate-name some_crate --crate-type lib -o $(TMPDIR)/some_crate \
		-Z codegen-backend=$(TMPDIR)/the_backend.dylib -Z unstable-options \
		--emit link,dep-info
	grep -x "This has been \"compiled\" successfully." $(TMPDIR)/libsome_crate.rlib
	# don't declare a dependency on the codegen backend if -Zbinary-dep-depinfo isn't used.
	grep -v "the_backend.dylib" $(TMPDIR)/some_crate.d
	
	$(RUSTC) some_crate.rs --crate-name some_crate --crate-type lib -o $(TMPDIR)/some_crate \
		-Z codegen-backend=$(TMPDIR)/the_backend.dylib -Z unstable-options \
		--emit link,dep-info -Zbinary-dep-depinfo
	grep -x "This has been \"compiled\" successfully." $(TMPDIR)/libsome_crate.rlib
	# but declare a dependency on the codegen backend if -Zbinary-dep-depinfo it used.
	grep "the_backend.dylib" $(TMPDIR)/some_crate.d
