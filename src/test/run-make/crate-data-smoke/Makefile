-include ../tools.mk

all:
	[ `$(RUSTC) --print-crate-name crate.rs` = "foo" ]
	[ `$(RUSTC) --print-file-name crate.rs` = "$(call BIN,foo)" ]
	[ `$(RUSTC) --print-file-name --crate-type=lib \
		--test crate.rs` = "$(call BIN,foo)" ]
	[ `$(RUSTC) --print-file-name --test lib.rs` = "$(call BIN,mylib)" ]
	$(RUSTC) --print-file-name lib.rs
	$(RUSTC) --print-file-name rlib.rs
