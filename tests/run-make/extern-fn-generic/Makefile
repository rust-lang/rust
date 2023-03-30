include ../tools.mk

all: $(call NATIVE_STATICLIB,test)
	$(RUSTC) testcrate.rs
	$(RUSTC) test.rs
	$(call RUN,test) || exit 1
