#![feature(rustc_attrs)]

trait Foo<A> {
    fn foo(self);
}

#[rustc_on_unimplemented = "an impl did not match: {A} {B} {C}"]
impl<A, B, C> Foo<A> for (A, B, C) {
    fn foo(self) {}
}

fn main() {
    Foo::<usize>::foo((1i32, 1i32, 1i32));
    //~^ ERROR trait `Foo<usize>` is not implemented for `(i32, i32, i32)`
}
