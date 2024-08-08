#![feature(no_sanitize)]
#![feature(stmt_expr_attributes)]
#![deny(unused_attributes)]
#![allow(dead_code)]

fn invalid() {
    #[no_sanitize(memory)] //~ ERROR attribute should be applied to a function definition
    {
        1
    };
}

#[no_sanitize(memory)] //~ ERROR attribute should be applied to a function definition
type InvalidTy = ();

#[no_sanitize(memory)] //~ ERROR attribute should be applied to a function definition
mod invalid_module {}

fn main() {
    let _ = #[no_sanitize(memory)] //~ ERROR attribute should be applied to a function definition
    (|| 1);
}

#[no_sanitize(memory)] //~ ERROR attribute should be applied to a function definition
struct F;

#[no_sanitize(memory)] //~ ERROR attribute should be applied to a function definition
impl F {
    #[no_sanitize(memory)]
    fn valid(&self) {}
}

#[no_sanitize(memory)]
fn valid() {}
