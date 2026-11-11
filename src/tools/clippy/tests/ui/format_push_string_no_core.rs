//@check-pass
#![warn(clippy::format_push_string)]
#![no_std]
#![feature(no_core)]
#![no_core]

extern crate alloc;

use alloc::format;
use alloc::string::String;

fn foo(string: &mut String) {
    // can't suggest even `core::fmt::Write` because of `#![no_core]`
    string.push_str(&format!("{:?}", 1234));
}
