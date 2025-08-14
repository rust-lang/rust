//@ check-pass

trait Foo {
    type Bar
    where
        Self: Sized;
}

fn foo(_: &dyn Foo<Bar = ()>) {}
//~^ WARN: unnecessary associated type bound for dyn-incompatible associated type

#[allow(unused_associated_type_bounds)]
fn bar(_: &dyn Foo<Bar = ()>) {}

fn main() {}
