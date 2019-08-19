// edition:2018
#![feature(async_await)]
#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

use std::io::Error;

fn make_unit() -> Result<(), Error> {
    Ok(())
}

fn main() {
    let fut = async {
        make_unit()?; //~ ERROR type annotations needed

        Ok(())
    };
}
