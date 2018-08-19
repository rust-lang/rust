-include ../tools.mk

ifeq (musl,$(findstring musl,$(TARGET)))
all: skip
else
all: test
endif

test: foo
	$(call RUN,foo)

skip:
	echo "expected failure"

foo: foo.rs $(call NATIVE_STATICLIB,foo)
	$(RUSTC) $< -lfoo $(EXTRACXXFLAGS)

$(TMPDIR)/libfoo.o: foo.cpp
	$(call COMPILE_OBJ_CXX,$@,$<)

.PHONY: all test skip
