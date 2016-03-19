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
