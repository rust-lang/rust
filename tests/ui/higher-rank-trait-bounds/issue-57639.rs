// Regression test for #57639:
//
// In the move to universes, this test stopped working. The problem
// was that when the trait solver was asked to prove `for<'a> T::Item:
// Foo<'a>` as part of WF checking, it wound up "eagerly committing"
// to the where clause, which says that `T::Item: Foo<'a>`, but it
// should instead have been using the bound found in the trait
// declaration. Pre-universe, this used to work out ok because we got
// "eager errors" due to the leak check.
//
// See [this comment on GitHub][c] for more details.
//
// check-pass
//
// [c]: https://github.com/rust-lang/rust/issues/57639#issuecomment-455685861

trait Foo<'a> {}

trait Bar {
    type Item: for<'a> Foo<'a>;
}

fn foo<'a, T>(_: T)
where
    T: Bar,
    T::Item: Foo<'a>,
{}

fn main() { }
