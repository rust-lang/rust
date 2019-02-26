-include ../tools.mk

# ignore-musl

all: foo
	$(call RUN,foo)

foo: foo.rs $(call NATIVE_STATICLIB,foo)
	$(RUSTC) $< -lfoo $(EXTRACXXFLAGS)

$(TMPDIR)/libfoo.o: foo.cpp
	$(call COMPILE_OBJ_CXX,$@,$<)
