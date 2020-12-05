// run-pass
fn main() {
    struct X;
    trait Foo<T> {
        fn foo(&self) where (T, Option<T>): Ord {}
        fn bar(&self, x: &Option<T>) -> bool
        where Option<T>: Ord { *x < *x }
    }
    impl Foo<X> for () {}
    let _ = &() as &dyn Foo<X>;
}
