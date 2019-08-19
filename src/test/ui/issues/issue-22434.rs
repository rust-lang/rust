pub trait Foo {
    type A;
}

type I<'a> = &'a (dyn Foo + 'a);
//~^ ERROR the value of the associated type `A` (from the trait `Foo`) must be specified

fn main() {}
