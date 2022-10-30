include ../tools.mk

all: $(call NATIVE_STATICLIB,ctest)
	$(RUSTC) test.rs
	$(call RUN,test) || exit 1
