// run-pass
#![allow(unused_variables)]
// Test that you can supply `&F` where `F: Fn()`.

#![feature(lang_items)]

fn a<F:Fn() -> i32>(f: F) -> i32 {
    f()
}

fn b(f: &dyn Fn() -> i32) -> i32 {
    a(f)
}

fn c<F:Fn() -> i32>(f: &F) -> i32 {
    a(f)
}

fn main() {
    let z: isize = 7;

    let x = b(&|| 22);
    assert_eq!(x, 22);

    let x = c(&|| 22);
    assert_eq!(x, 22);
}
