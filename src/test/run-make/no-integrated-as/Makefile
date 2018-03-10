-include ../tools.mk

all:
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
	$(RUSTC) hello.rs -C no_integrated_as
	$(call RUN,hello)
endif
