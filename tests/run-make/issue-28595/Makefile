include ../tools.mk

all: $(call NATIVE_STATICLIB,a) $(call NATIVE_STATICLIB,b)
	$(RUSTC) a.rs
	$(RUSTC) b.rs
	$(call RUN,b)
