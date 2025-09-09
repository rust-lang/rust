//@ edition: 2021
#![feature(c_variadic)]
#![crate_type = "lib"]

async unsafe extern "C" fn fn_cannot_be_async(x: isize, ...) {}
//~^ ERROR functions cannot be both `async` and C-variadic
//~| ERROR hidden type for `impl Future<Output = ()>` captures lifetime that does not appear in bounds

struct S;

impl S {
    async unsafe extern "C" fn method_cannot_be_async(x: isize, ...) {}
    //~^ ERROR functions cannot be both `async` and C-variadic
    //~| ERROR hidden type for `impl Future<Output = ()>` captures lifetime that does not appear in bounds
}
