//@ edition:2018
//@ proc-macro: same-res-ambigious-extern-macro.rs

#[macro_use] // this imports the `RustEmbed` macro with `pub(crate)` visibility
extern crate same_res_ambigious_extern_macro;
// this imports the same `RustEmbed` macro with `pub` visibility
pub use same_res_ambigious_extern_macro::*;

pub trait RustEmbed {}

pub use RustEmbed as Embed;
