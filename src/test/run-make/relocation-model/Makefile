-include ../tools.mk

all: others
	$(RUSTC) -C relocation-model=dynamic-no-pic foo.rs
	$(call RUN,foo)

	$(RUSTC) -C relocation-model=default foo.rs
	$(call RUN,foo)

	$(RUSTC) -C relocation-model=default --crate-type=dylib foo.rs
	$(RUSTC) -C relocation-model=dynamic-no-pic --crate-type=dylib foo.rs

ifdef IS_MSVC
# FIXME(#28026)
others:
else
others:
	$(RUSTC) -C relocation-model=static foo.rs
	$(call RUN,foo)
	$(RUSTC) -C relocation-model=static --crate-type=dylib foo.rs
endif
