// run-rustfix

#![deny(no_mangle_generic_items)]

#[no_mangle]
pub fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

#[no_mangle]
pub extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

#[no_mangle]
pub fn baz(x: &i32) -> &i32 { x }

#[no_mangle]
pub fn qux<'a>(x: &'a i32) -> &i32 { x }

fn main() {}
