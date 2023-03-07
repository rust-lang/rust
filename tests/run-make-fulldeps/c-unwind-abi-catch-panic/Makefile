include ../tools.mk

all: $(call NATIVE_STATICLIB,add)
	$(RUSTC) main.rs
	$(call RUN,main) || exit 1
