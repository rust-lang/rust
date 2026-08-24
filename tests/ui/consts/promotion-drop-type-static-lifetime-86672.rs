// Regression test for <https://github.com/rust-lang/rust/issues/86672>.
// Borrowing an array of a Drop type in a const used to fail with E0493 and E0716
// unless the borrow went through another const.
//@ check-pass

#![allow(dead_code)]

pub struct Foo<'a, B: ?Sized>(&'a B);

struct Bar;
impl Drop for Bar {
    fn drop(&mut self) {}
}

// These always worked.
const BAR0: Bar = Bar;
const BAR1: &'static [Bar] = &[Bar];
const BAR2: Foo<'static, [Bar]> = Foo(BAR1);
// These used to fail.
const BAR3: Foo<'static, [Bar]> = Foo(&[Bar]);
const BAR4: Foo<'static, [Bar]> = Foo(&[Bar] as &'static [Bar]);

fn main() {}
