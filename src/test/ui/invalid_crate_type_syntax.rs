// regression test for issue 16974
#![crate_type(lib)]  //~ ERROR `crate_type` requires a value

fn my_lib_fn() {}

fn main() {}
