// check-pass

#![feature(unboxed_closures, fn_traits)]

fn main() {
    let k = |x: i32| { x + 1 };
    Fn::call(&k, (0,));
}
