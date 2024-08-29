//@ check-pass

#![deny(non_snake_case)]

#[no_mangle]
pub extern "C" fn SparklingGenerationForeignFunctionInterface() {} // OK

pub struct Foo;

impl Foo {
    #[no_mangle]
    pub extern "C" fn SparklingGenerationForeignFunctionInterface() {} // OK
}

fn main() {}
