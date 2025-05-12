//@ known-bug: #132430

//@ compile-flags: --crate-type=lib
//@ edition: 2018
#![feature(cmse_nonsecure_entry)]
struct Test;

impl Test {
    pub async unsafe extern "C-cmse-nonsecure-entry" fn test(val: &str) {}
}
