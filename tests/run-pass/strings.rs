#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn empty() -> &'static str {
    ""
}

#[miri_run]
fn hello() -> &'static str {
    "Hello, world!"
}

#[miri_run]
fn hello_bytes() -> &'static [u8; 13] {
    b"Hello, world!"
}

#[miri_run]
fn hello_bytes_fat() -> &'static [u8] {
    b"Hello, world!"
}

fn main() {}
