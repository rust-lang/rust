// aux-build:proc_macros.rs
#![warn(clippy::mem_replace_with_default)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    let s = &mut String::from("foo");
    inline!(std::mem::replace($s, Default::default()));
    external!(std::mem::replace($s, Default::default()));
}
