// check-pass
// known-bug: #74629

// Should fail. The `0` and `1` impls overlap, violating coherence. Eg, with
// `T = Test, F = ()`, all bounds are true, making both impls applicable.
// `Test: Fold<Nil>`, `Test: Fold<()>` are true because of `2`.
// `Is<Test>: NotNil` is true because of `auto trait` and lack of negative impl.

#![feature(negative_impls)]
#![feature(auto_traits)]

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

    #[allow(suspicious_auto_trait_impls)]
    impl !NotNil for Is<Nil> {}
}

fn main() {}
