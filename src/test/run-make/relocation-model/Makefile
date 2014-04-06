-include ../tools.mk

all:
	$(RUSTC) -C relocation-model=dynamic-no-pic foo.rs
	$(call RUN,foo)

	$(RUSTC) -C relocation-model=default foo.rs
	$(call RUN,foo)

	$(RUSTC) -C relocation-model=static foo.rs
	$(call RUN,foo)

	$(RUSTC) -C relocation-model=default --crate-type=dylib foo.rs
	$(RUSTC) -C relocation-model=static --crate-type=dylib foo.rs
	$(RUSTC) -C relocation-model=dynamic-no-pic --crate-type=dylib foo.rs
