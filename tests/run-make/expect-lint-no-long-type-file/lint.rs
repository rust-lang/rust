#![expect(improper_ctypes_definitions, dead_code)]

extern "C" fn foo(_: Option<Option<Option<Option<Option<Option<Option<Option<i32>>>>>>>>) {}

fn main() {}
