#![feature(negative_impls)]
#![feature(optin_builtin_traits)]
struct Nil;
struct Cons<H>(H);
struct Test;

trait Fold<F> {}

impl<T, F> Fold<F> for Cons<T> // 0
where
    T: Fold<Nil>,
{}

impl<T, F> Fold<F> for Cons<T> // 1
where
    T: Fold<F>,
    private::Is<T>: private::NotNil,
{}

impl<F> Fold<F> for Test {} // 2

mod private {
    use crate::Nil;

    pub struct Is<T>(T);
    pub auto trait NotNil {}
    impl !NotNil for Is<Nil> {} //~ ERROR `private::NotNil` impls cannot
}

fn main() {}
