-include ../tools.mk

all:
	[ `$(RUSTC) --crate-id crate.rs` = "foo#0.10-pre" ]
	[ `$(RUSTC) --crate-name crate.rs` = "foo" ]
	[ `$(RUSTC) --crate-file-name crate.rs` = "foo" ]
	[ `$(RUSTC) --crate-file-name --lib --test crate.rs` = "foo" ]
	[ `$(RUSTC) --crate-file-name --test lib.rs` = "mylib" ]
	$(RUSTC) --crate-file-name lib.rs
	$(RUSTC) --crate-file-name rlib.rs
