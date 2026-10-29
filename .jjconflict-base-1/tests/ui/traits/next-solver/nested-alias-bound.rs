//@ compile-flags: -Znext-solver
//@ check-pass

trait A {
    type A: B;
}

trait B {
    type B: C;
}

trait C {}

fn needs_c<T: C>() {}

fn test<T: A>() {
    needs_c::<<T::A as B>::B>();
}

fn main() {}
