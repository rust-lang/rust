//@ revisions: unused_attr unused_cli
//@[unused_cli] compile-flags: -Dunused_imports
//@ check-pass

// This test purpose is to check that unused_imports lint isn't fired
// by rustdoc. Why would it? Because when rustdoc is running, it uses
// "everybody-loops" which replaces parts of code with "loop {}" to get
// huge performance improvements.

#![cfg_attr(unused_attr, deny(unused_imports))]

use std::fs::File;

pub fn f() {
    let _: File;
}
