//@ edition:2018

#![feature(c_variadic)]

async unsafe extern "C" fn multiple_named_lifetimes<'a, 'b>(_: u8, ...) {}
//~^ ERROR hidden type for `impl Future<Output = ()>` captures lifetime that does not appear in bounds

fn main() {}
