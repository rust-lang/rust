// It's not yet clear how '_ and GATs should interact.
// Forbid it for now but proper support might be added
// at some point in the future.

#![feature(anonymous_lifetime_in_impl_trait)]
trait Foo {
    type Item<'a>;
}

fn foo(x: &impl Foo<Item<'_> = u32>) { }
                       //~^ ERROR `'_` cannot be used here [E0637]

// Ok: the anonymous lifetime is bound to the function.
fn bar(x: &impl for<'a> Foo<Item<'a> = &'_ u32>) { }

// Ok: the anonymous lifetime is bound to the function.
fn baz(x: &impl for<'a> Foo<Item<'a> = &u32>) { }

fn main() {}
