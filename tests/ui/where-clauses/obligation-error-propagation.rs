//! Regression test for <https://github.com/rust-lang/rust/issues/34503>.
//! `(T, Option<T>)` falsly marked Option<T> as proved when T failed,
//! this made use of invalid Option<T> bound possible anywhere.
//@ run-pass

fn main() {
    struct X;
    trait Foo<T> {
        fn foo(&self) where (T, Option<T>): Ord {} //~ WARN methods `foo` and `bar` are never used
        fn bar(&self, x: &Option<T>) -> bool
        where Option<T>: Ord { *x < *x }
    }
    impl Foo<X> for () {}
    let _ = &() as &dyn Foo<X>;
}
