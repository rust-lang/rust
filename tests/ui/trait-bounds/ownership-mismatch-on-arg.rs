// #134805
mod needs_deref {
    #[derive(Clone, Copy, Debug)]
    struct Hello;

    trait Tr: Clone + Copy {}
    impl Tr for Hello {}

    fn foo<T: Tr, K: std::fmt::Debug>(_v: T, _w: T, _k: K) {}

    struct S;
    impl S {
        fn foo<K: std::fmt::Debug, T: Tr>(&self, _v: T, _w: T, _k: K) {}
    }

    fn bar() {
        let hellos = [Hello; 3];
        for hi in hellos.iter() {
            foo(hi, hi, hi); //~ ERROR: the trait bound `&needs_deref::Hello: needs_deref::Tr` is not satisfied
            S.foo(hi, hi, hi); //~ ERROR: the trait bound `&needs_deref::Hello: needs_deref::Tr` is not satisfied
        }
    }
}

mod needs_borrow {
    #[derive(Clone, Copy, Debug)]
    struct Hello;

    trait Tr: Clone + Copy {}
    impl Tr for &Hello {}

    fn foo<T: Tr, K: std::fmt::Debug>(_v: T, _w: T, _k: K) {}

    struct S;
    impl S {
        fn foo<T: Tr, K: std::fmt::Debug>(&self, _v: T, _w: T, _k: K) {}
    }

    fn bar() {
        let hellos = [Hello; 3];
        for hi in hellos {
            foo(hi, hi, hi); //~ ERROR: the trait bound `needs_borrow::Hello: needs_borrow::Tr` is not satisfied
            S.foo(hi, hi, hi); //~ ERROR: the trait bound `needs_borrow::Hello: needs_borrow::Tr` is not satisfied
        }
    }
}
fn main() {}
