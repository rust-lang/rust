-include ../tools.mk

ifdef IS_MSVC
# FIXME(#27979)
all:
else
all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(RUSTC) main.rs
	$(call RUN,main)
endif
