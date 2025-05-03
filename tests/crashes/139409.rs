//@ known-bug: #139409
//@ compile-flags: -Znext-solver=globally

fn main() {
    trait B<C> {}
    impl<C> B<C> for () {}
    trait D<C, E>: B<C> + B<E> {
        fn f(&self) {}
    }
    impl<C, E> D<C, E> for () {}
    (&() as &dyn D<&(), &()>).f()
}
