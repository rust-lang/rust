// aux-build: macro-2-reexport.rs

#![crate_name = "foo"]

extern crate macro_2_reexport;

// @has 'foo/macro.addr_of.html' '//*[@class="docblock type-decl"]' 'macro addr_of($place : expr) {'
pub use macro_2_reexport::addr_of;
