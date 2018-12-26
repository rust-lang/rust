// Tests that `transmute` cannot be called on type parameters.

use std::mem::transmute;

unsafe fn f<T>(x: T) {
    let _: i32 = transmute(x);
//~^ ERROR transmute called with types of different sizes
}

unsafe fn g<T>(x: (T, i32)) {
    let _: i32 = transmute(x);
//~^ ERROR transmute called with types of different sizes
}

unsafe fn h<T>(x: [T; 10]) {
    let _: i32 = transmute(x);
//~^ ERROR transmute called with types of different sizes
}

struct Bad<T> {
    f: T,
}

unsafe fn i<T>(x: Bad<T>) {
    let _: i32 = transmute(x);
//~^ ERROR transmute called with types of different sizes
}

enum Worse<T> {
    A(T),
    B,
}

unsafe fn j<T>(x: Worse<T>) {
    let _: i32 = transmute(x);
//~^ ERROR transmute called with types of different sizes
}

unsafe fn k<T>(x: Option<T>) {
    let _: i32 = transmute(x);
//~^ ERROR transmute called with types of different sizes
}

fn main() {}
