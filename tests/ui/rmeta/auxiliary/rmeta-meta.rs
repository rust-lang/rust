//@ no-prefer-dynamic
//@ compile-flags: --emit=metadata

#![crate_type="rlib"]

pub struct Foo {
    pub field: i32,
}

pub fn missing_optimized_mir() {
    println!("indeed");
}
