// regression test for issue 16974
#![crate_type(lib)]  //~ ERROR malformed `crate_type` attribute input

fn my_lib_fn() {}

fn main() {}
