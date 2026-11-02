//@ check-pass

#![feature(fn_delegation)]

mod simple_test {
    trait Trait {
        fn foo(&self) -> Self;
    }

    struct S;
    impl Trait for S {
        fn foo(&self) -> S { S }
    }

    struct W { s: S }
    impl Trait for W {
        reuse Trait::foo { self.s }
    }

    impl W {
        reuse Trait::foo { self.s }
    }

    struct W1(S);
    impl Trait for W1 {
        reuse Trait::foo { self.0 }
    }

    impl W1 {
        reuse Trait::foo { self.0 }
    }
}

mod rfc {
    struct BTreeMap<K, V> {
        _inner: std::collections::HashMap<K, V>
    }

    impl<K, V> Clone for BTreeMap<K, V> {
        fn clone(&self) -> Self {
            panic!()
        }

        fn clone_from(&mut self, source: &Self) {
            panic!()
        }
    }

    struct BTreeSet<T> {
        map: BTreeMap<T, ()>
    }

    reuse impl<T: Clone> Clone for BTreeSet<T> { self.map }
}

fn main() {}
