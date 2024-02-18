//@ force-host
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::Literal;

fn test() {
    Literal::byte_character(b'a'); //~ ERROR use of unstable library feature 'proc_macro_byte_character'
}
