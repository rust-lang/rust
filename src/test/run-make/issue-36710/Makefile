# ignore-cross-compile $(call RUN,foo) expects to run the target executable natively
#                              so it won't work with remote-test-server
# ignore-none no-std is not supported
# ignore-musl FIXME: this makefile needs teaching how to use a musl toolchain
#                    (see dist-i586-gnu-i586-i686-musl Dockerfile)

include ../../run-make-fulldeps/tools.mk

all: foo
	$(call RUN,foo)

foo: foo.rs $(call NATIVE_STATICLIB,foo)
	$(RUSTC) $< -lfoo $(EXTRARSCXXFLAGS) --target $(TARGET)

$(TMPDIR)/libfoo.o: foo.cpp
	$(call COMPILE_OBJ_CXX,$@,$<)
