-include ../tools.mk

all:
	[ `$(RUSTC) --print crate-name crate.rs` = "foo" ]
	[ `$(RUSTC) --print file-names crate.rs` = "$(call BIN,foo)" ]
	[ `$(RUSTC) --print file-names --crate-type=lib \
		--test crate.rs` = "$(call BIN,foo)" ]
	[ `$(RUSTC) --print file-names --test lib.rs` = "$(call BIN,mylib)" ]
	$(RUSTC) --print file-names lib.rs
	$(RUSTC) --print file-names rlib.rs
