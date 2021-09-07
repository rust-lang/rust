# ignore-cross-compile -- compiling C++ code does not work well when cross-compiling

# This test case makes sure that native libraries are linked with --whole-archive semantics
# when the `-bundle,+whole-archive` modifiers are applied to them.
#
# The test works by checking that the resulting executables produce the expected output,
# part of which is emitted by otherwise unreferenced C code. If +whole-archive didn't work
# that code would never make it into the final executable and we'd thus be missing some
# of the output.

-include ../../run-make-fulldeps/tools.mk

all: $(TMPDIR)/$(call BIN,directly_linked) $(TMPDIR)/$(call BIN,indirectly_linked) $(TMPDIR)/$(call BIN,indirectly_linked_via_attr)
	$(call RUN,directly_linked) | $(CGREP) 'static-initializer.directly_linked.'
	$(call RUN,indirectly_linked) | $(CGREP) 'static-initializer.indirectly_linked.'
	$(call RUN,indirectly_linked_via_attr) | $(CGREP) 'static-initializer.native_lib_in_src.'

# Native lib linked directly into executable
$(TMPDIR)/$(call BIN,directly_linked): $(call NATIVE_STATICLIB,c_static_lib_with_constructor)
	$(RUSTC) directly_linked.rs -Z unstable-options -l static:+whole-archive=c_static_lib_with_constructor

# Native lib linked into RLIB via `-l static:-bundle,+whole-archive`, RLIB linked into executable
$(TMPDIR)/$(call BIN,indirectly_linked): $(TMPDIR)/librlib_with_cmdline_native_lib.rlib
	$(RUSTC) indirectly_linked.rs

# Native lib linked into RLIB via #[link] attribute, RLIB linked into executable
$(TMPDIR)/$(call BIN,indirectly_linked_via_attr): $(TMPDIR)/libnative_lib_in_src.rlib
	$(RUSTC) indirectly_linked_via_attr.rs

# Native lib linked into rlib with via commandline
$(TMPDIR)/librlib_with_cmdline_native_lib.rlib: $(call NATIVE_STATICLIB,c_static_lib_with_constructor)
	$(RUSTC) rlib_with_cmdline_native_lib.rs -Z unstable-options --crate-type=rlib -l static:-bundle,+whole-archive=c_static_lib_with_constructor

# Native lib linked into rlib via `#[link()]` attribute on extern block.
$(TMPDIR)/libnative_lib_in_src.rlib: $(call NATIVE_STATICLIB,c_static_lib_with_constructor)
	$(RUSTC) native_lib_in_src.rs --crate-type=rlib

$(TMPDIR)/libc_static_lib_with_constructor.o: c_static_lib_with_constructor.cpp
	$(call COMPILE_OBJ_CXX,$@,$<)
