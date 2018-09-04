#![feature(min_const_fn)]

const fn foo1() {}
const fn foo2(x: i32) -> i32 { x }

fn main() {
    let x: &'static () = &foo1();
    let y: &'static i32 = &foo2(42); //~ ERROR does not live long enough
}