# ignore-riscv64 $(call RUN,foo) expects to run the target executable natively
#                              so it won't work with remote-test-server
# ignore-arm Another build using remote-test-server
# ignore-none no-std is not supported

include ../../run-make-fulldeps/tools.mk

all: foo
	$(call RUN,foo)

foo: foo.rs $(call NATIVE_STATICLIB,foo)
	$(RUSTC) $< -lfoo $(EXTRARSCXXFLAGS) --target $(TARGET)

$(TMPDIR)/libfoo.o: foo.cpp
	$(call COMPILE_OBJ_CXX,$@,$<)
