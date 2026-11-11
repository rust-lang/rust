// Make sure we are using the right binder vars when expanding
// `for<'a> Foo<'a>` to `for<'a> Bar<'a>`.

//@ check-pass

#![feature(trait_alias)]

trait Bar<'a> {}

trait Foo<'a> = Bar<'a>;

fn test2(_: &(impl for<'a> Foo<'a> + ?Sized)) {}

fn test(x: &dyn for<'a> Foo<'a>) {
    test2(x);
}

fn main() {}
