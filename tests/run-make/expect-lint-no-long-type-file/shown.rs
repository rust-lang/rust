#![warn(improper_ctypes_definitions)]

extern "C" fn foo(_: Option<Option<Option<Option<Option<Option<Option<Option<i32>>>>>>>>) {}

fn main() {}
