// https://github.com/rust-lang/rust/issues/22434
pub trait Foo {
    type A;
}

type I<'a> = &'a (dyn Foo + 'a);
//~^ ERROR the value of the associated type `A` in `Foo` must be specified

fn main() {}
