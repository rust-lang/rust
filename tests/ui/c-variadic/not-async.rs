//@ edition: 2021
#![feature(c_variadic)]
#![crate_type = "lib"]

async unsafe extern "C" fn cannot_be_async(x: isize, ...) {}
//~^ ERROR functions cannot be both `async` and C-variadic
//~| ERROR hidden type for `impl Future<Output = ()>` captures lifetime that does not appear in bounds
