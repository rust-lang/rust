include ../tools.mk

all: $(call NATIVE_STATICLIB,foo)
	$(RUSTC) main.rs
	$(call RUN,main)
