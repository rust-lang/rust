//@ compile-flags: -Zunstable-options --error-format=human-unicode --color=always
//@ only-linux
// Ensure that when we have a type error where both types have the same textual representation, the
// diagnostic machinery highlights the clarifying comment that comes after in parentheses.
trait Foo: Copy + ToString {}

impl<T: Copy + ToString> Foo for T {}

fn hide<T: Foo>(x: T) -> impl Foo {
    x
}

fn main() {
    let mut x = (hide(0_u32), hide(0_i32));
    x = (x.1, x.0);
}
