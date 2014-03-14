-include ../tools.mk

all: $(call STATICLIB,foo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call RUN,bar)
