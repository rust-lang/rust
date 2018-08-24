#![feature(unboxed_closures, fn_traits, rustc_attrs)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let k = |x: i32| { x + 1 };
    Fn::call(&k, (0,));
}
