//! Checks that `link_section` can be used on foreign statics.
#![crate_type = "lib"]

//@ edition:2024
//@ check-pass
// It used to check for a warning on foreign statics (see
// <https://github.com/rust-lang/rust/issues/136220>). Now the attribute is
// accepted.
unsafe extern "C" {
    #[unsafe(link_section = "__DATA,__buffer")]
    pub static mut a: [u8; 1024];
}
