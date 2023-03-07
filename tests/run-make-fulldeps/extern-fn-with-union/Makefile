include ../tools.mk

all: $(call NATIVE_STATICLIB,ctest)
	$(RUSTC) testcrate.rs
	$(RUSTC) test.rs
	$(call RUN,test) || exit 1
