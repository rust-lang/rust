-include ../tools.mk

all:
	[ `$(RUSTC) --print crate-name crate.rs` = "foo" ]
	[ `$(RUSTC) --print file-names crate.rs` = "$(call BIN,foo)" ]
	[ `$(RUSTC) --print file-names --crate-type=lib \
		--test crate.rs` = "$(call BIN,foo)" ]
	[ `$(RUSTC) --print file-names --test lib.rs` = "$(call BIN,mylib)" ]
	[ "`$(RUSTC) --emit=dep-info,metadata --print file-names crate.rs`" = "" ] # Shouldn't print the binary name.
	$(RUSTC) --print file-names lib.rs
	$(RUSTC) --print file-names rlib.rs
