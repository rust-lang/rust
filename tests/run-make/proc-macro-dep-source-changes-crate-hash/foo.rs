// Consumer crate. Byte-identical across all invocations of the test;
// only the tokens spliced in by `#[derive(ChangingDerive)]` change between
// builds, driven by which version of the proc-macro is on disk.

#![crate_type = "rlib"]

extern crate changing_macro;

use changing_macro::ChangingDerive;

#[derive(ChangingDerive)]
pub struct Foo;

pub fn answer() -> u32 {
    ANSWER
}
