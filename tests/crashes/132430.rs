//@ known-bug: #132430

//@compile-flags: --edition=2018 --crate-type=lib
#![feature(cmse_nonsecure_entry)]
struct Test;

impl Test {
    pub async unsafe extern "C-cmse-nonsecure-entry" fn test(val: &str) {}
}
