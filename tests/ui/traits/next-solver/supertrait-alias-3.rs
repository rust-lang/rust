//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/171>.
// Exercises a case where structural equality is insufficient when replacing projections in a dyn's
// bounds. In this case, the bound will contain `<Self as Super<<i32 as Mirror>:Assoc>::Assoc`, but
// the existential projections from the dyn will have `<Self as Super<i32>>::Assoc` because as an
// optimization we eagerly normalize aliases in goals.

trait Other<T> {}
impl<T> Other<T> for T {}

trait Super<T> {
    type Assoc;
}

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

trait Foo<A, B>: Super<<A as Mirror>::Assoc, Assoc = A> {
    type FooAssoc: Other<<Self as Super<<A as Mirror>::Assoc>>::Assoc>;
}

fn is_foo<F: Foo<T, U> + ?Sized, T, U>() {}

fn main() {
    is_foo::<dyn Foo<i32, u32, FooAssoc = i32>, _, _>();
}
