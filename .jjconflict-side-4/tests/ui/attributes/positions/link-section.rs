//! Checks that `link_section` cannot be used on foreign statics.
#![crate_type = "lib"]

//@ edition:2024
//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/136220>.
unsafe extern "C" {
    #[unsafe(link_section = "__DATA,__buffer")] //~ WARN attribute cannot be used on foreign statics
    //~| WARN previously accepted
    pub static mut a: [u8; 1024];
}
