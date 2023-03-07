#![crate_type = "lib"]

extern fn none_fn(x: bool) -> i32;
//~^ ERROR free function without a body
extern "C" fn c_fn(x: bool) -> i32;
//~^ ERROR free function without a body
