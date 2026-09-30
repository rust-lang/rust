use std::hint::black_box;
use std::mem;

fn add_one(x: i32) -> i32 {
    x + 1
}

#[inline(never)]
fn call_with_mismatch(f: fn(i32) -> i32) {
    let g: fn(i32, i32) -> i32 = unsafe { mem::transmute(f) };
    let res = g(1, 2);
    assert_eq!(res, 2);
}

fn main() {
    call_with_mismatch(black_box(add_one));
}
