//! Regression test for <https://github.com/rust-lang/rust/issues/24308>.
//! Ensure we're not ICE'ing on calls to trait method through Self in method body.
//! (while implementing for struct with explicit lifetimes)
//@ run-pass

pub trait Foo {
    fn method1() {}
    fn method2();
}

struct Slice<'a, T: 'a>(#[allow(dead_code)] &'a [T]);

impl<'a, T: 'a> Foo for Slice<'a, T> {
    fn method2() {
        <Self as Foo>::method1();
    }
}

fn main() {
    <Slice<()> as Foo>::method2();
}
