//@ run-pass

#![allow(incomplete_features)]
#![feature(unsized_fn_params)]

pub fn f0(_f: dyn FnOnce()) {}
pub fn f1(_s: str) {}
pub fn f2(_x: i32, _y: [i32]) {}

fn main() {
    let foo = "foo".to_string().into_boxed_str();
    f1(*foo);
}
