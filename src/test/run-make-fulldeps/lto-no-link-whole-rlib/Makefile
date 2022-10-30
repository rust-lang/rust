include ../tools.mk

all: $(call NATIVE_STATICLIB,foo) $(call NATIVE_STATICLIB,bar)
	$(RUSTC) lib1.rs
	$(RUSTC) lib2.rs
	$(RUSTC) main.rs -Clto
	$(call RUN,main)

