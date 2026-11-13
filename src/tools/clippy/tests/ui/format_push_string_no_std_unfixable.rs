//@no-rustfix
#![warn(clippy::format_push_string)]
#![no_std]

extern crate alloc;

use alloc::format;
use alloc::string::String;

fn foo(string: &mut String) {
    string.push_str(&format!("{:?}", 1234));
    //~^ format_push_string
}
