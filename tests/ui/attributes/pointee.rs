#![feature(pointee)]
#![feature(stmt_expr_attributes)]
#![feature(derive_smart_pointer)]
#![deny(unused_attributes)]
#![allow(dead_code)]

fn invalid() {
    #[pointee] //~ ERROR attribute should be applied to a function definition
    {
        1
    };
}

#[pointee] //~ ERROR attribute should be applied to a function definition
type InvalidTy = ();

#[pointee] //~ ERROR attribute should be applied to a function definition
mod invalid_module {}

fn main() {
    let _ = #[pointee] //~ ERROR attribute should be applied to a function definition
    (|| 1);
}

#[pointee] //~ ERROR attribute should be applied to a function definition
struct F;

#[pointee] //~ ERROR attribute should be applied to a function definition
impl F {
    #[pointee]
    fn valid(&self) {}
}

#[pointee]
fn valid() {}
