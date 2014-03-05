-include ../tools.mk

all: $(call STATICLIB,foo) $(call STATICLIB,bar)
	$(RUSTC) main.rs
	$(call RUN,main)

