// It's not yet clear how '_ and GATs should interact.
// Forbid it for now but proper support might be added
// at some point in the future.

#![feature(generic_associated_types)]

trait Foo {
    type Item<'a>;
}

fn foo(x: &impl Foo<Item<'_> = u32>) { }
                       //~^ ERROR missing lifetime specifier

fn bar(x: &impl for<'a> Foo<Item<'a> = &'_ u32>) { }
                                      //~^ ERROR missing lifetime specifier

fn main() {}
