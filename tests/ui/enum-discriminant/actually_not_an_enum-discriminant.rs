//@ run-pass
#![feature(core_intrinsics)]

use std::intrinsics::discriminant_value;

struct Zst;

struct Struct {
    _a: u32,
}

union Union {
    _a: u32,
}

fn check(v: u8) {
    assert_eq!(v, 0);
}

pub fn generic<T>()
where
    for<'a> T: Fn(&'a isize),
{
    let v: Vec<T> =  Vec::new();
    let _: u8 = discriminant_value(&v);
}

fn main() {
    // check that we use `u8` as the discriminant value
    // for everything that is not an enum.
    check(discriminant_value(&true));
    check(discriminant_value(&'a'));
    check(discriminant_value(&7));
    check(discriminant_value(&7.0));
    check(discriminant_value(&Zst));
    check(discriminant_value(&Struct { _a: 7 }));
    check(discriminant_value(&Union { _a: 7 }));
    check(discriminant_value(&[7, 77]));
    check(discriminant_value(&(7 as *const ())));
    check(discriminant_value(&(7 as *mut ())));
    check(discriminant_value(&&7));
    check(discriminant_value(&&mut 7));
    check(discriminant_value(&check));
    let fn_ptr: fn(u8) = check;
    check(discriminant_value(&fn_ptr));
    let hrtb: for<'a> fn(&'a str) -> &'a str = |x| x;
    check(discriminant_value(&hrtb));
    check(discriminant_value(&(7, 77, 777)));
}
