#![feature(no_sanitize)]
#![feature(stmt_expr_attributes)]
#![deny(unused_attributes)]
#![allow(dead_code)]

fn invalid() {
    #[no_sanitize(memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
    {
        1
    };
}

#[no_sanitize(memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
type InvalidTy = ();

#[no_sanitize(memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
mod invalid_module {}

fn main() {
    let _ = #[no_sanitize(memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
    (|| 1);
}

#[no_sanitize(memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
struct F;

#[no_sanitize(memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
impl F {
    #[no_sanitize(memory)]
    fn valid(&self) {}
}

#[no_sanitize(address, memory)] //~ ERROR `#[no_sanitize(memory)]` should be applied to a function
static INVALID : i32 = 0;

#[no_sanitize(memory)]
fn valid() {}

#[no_sanitize(address)]
static VALID : i32 = 0;

#[no_sanitize("address")]
//~^ ERROR `#[no_sanitize(...)]` should be applied to a function
//~| ERROR invalid argument for `no_sanitize`
static VALID2 : i32 = 0;
