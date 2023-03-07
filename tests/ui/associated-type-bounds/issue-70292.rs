// check-pass

#![feature(associated_type_bounds)]

fn foo<F>(_: F)
where
    F: for<'a> Trait<Output: 'a>,
{
}

trait Trait {
    type Output;
}

impl<T> Trait for T {
    type Output = ();
}

fn main() {
    foo(());
}
