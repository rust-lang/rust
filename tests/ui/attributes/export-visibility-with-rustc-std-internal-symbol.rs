// This test verfies that `#[export_visibility = ...]` will report an error
// when applied to an item that also has `#[rustc_std_internal_symbol]`
// attribute.
#![feature(export_visibility)]
#![feature(rustc_attrs)]
#[export_visibility = "target_default"]
//~^ERROR: #[export_visibility = ...]` cannot be used on internal language items
#[rustc_std_internal_symbol]
pub static TESTED_STATIC: [u8; 6] = *b"foobar";

fn main() {}
