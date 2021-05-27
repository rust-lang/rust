// edition:2018
#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete

use std::io::Error;

fn make_unit() -> Result<(), Error> {
    Ok(())
}

fn main() {
    let fut = async {
        make_unit()?;

        Ok(()) //~ ERROR type annotations needed
    };
}
