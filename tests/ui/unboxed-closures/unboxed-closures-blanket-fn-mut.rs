//@ run-pass
#![allow(unused_variables)]
// Test that you can supply `&F` where `F: FnMut()`.

#![feature(lang_items)]

fn a<F:FnMut() -> i32>(mut f: F) -> i32 {
    f()
}

fn b(f: &mut dyn FnMut() -> i32) -> i32 {
    a(f)
}

fn c<F:FnMut() -> i32>(f: &mut F) -> i32 {
    a(f)
}

fn main() {
    let z: isize = 7;

    let x = b(&mut || 22);
    assert_eq!(x, 22);

    let x = c(&mut || 22);
    assert_eq!(x, 22);
}
