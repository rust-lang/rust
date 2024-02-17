//@ build-pass
#![crate_type = "lib"]
#![feature(unsized_fn_params)]

pub fn f(mut x: [i32]) {
    x[0] = 1;
}
