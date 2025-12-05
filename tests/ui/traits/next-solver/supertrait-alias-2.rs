//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/171>.
// Tests that we don't try to replace `<T as Other>::Assoc` when replacing projections in the
// required bounds for `dyn Foo`, b/c `T` is not relevant to the dyn type, which we were
// encountering when walking through the elaborated supertraits of `dyn Foo`.

trait Other<X> {}

trait Foo<T: Foo<T>>: Other<<T as Foo<T>>::Assoc> {
    type Assoc;
}

impl<T> Foo<T> for T {
    type Assoc = ();
}

impl<T: ?Sized> Other<()> for T {}

fn is_foo<T: Foo<()> + ?Sized>() {}

fn main() {
    is_foo::<dyn Foo<(), Assoc = ()>>();
}
