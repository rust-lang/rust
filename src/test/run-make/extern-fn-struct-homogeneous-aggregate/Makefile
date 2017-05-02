-include ../tools.mk

all: $(call NATIVE_STATICLIB,test)
	$(RUSTC) test.rs
	$(call RUN,test) || exit 1
