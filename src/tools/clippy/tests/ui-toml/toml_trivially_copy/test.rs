//@normalize-stderr-test: "\(\d+ byte\)" -> "(N byte)"
//@normalize-stderr-test: "\(limit: \d+ byte\)" -> "(limit: N byte)"
//@no-rustfix
#![warn(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::needless_pass_by_ref_mut)]

#[derive(Copy, Clone)]
struct Foo(u8);

#[derive(Copy, Clone)]
struct Bar(u32);

fn good(a: &mut u32, b: u32, c: &Bar, d: &u32) {}

fn bad(x: &u16, y: &Foo) {}
//~^ trivially_copy_pass_by_ref
//~| trivially_copy_pass_by_ref

fn main() {
    let (mut a, b, c, d, x, y) = (0, 0, Bar(0), 0, 0, Foo(0));
    good(&mut a, b, &c, &d);
    bad(&x, &y);
}
