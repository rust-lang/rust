// regression test for issue 16974
#![crate_type(lib)]  //~ ERROR attribute must be of the form

fn my_lib_fn() {}

fn main() {}
