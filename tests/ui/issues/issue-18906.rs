//@ check-pass
#![allow(dead_code)]

pub trait Borrow<Borrowed: ?Sized> {
        fn borrow(&self) -> &Borrowed;
}

impl<T: Sized> Borrow<T> for T {
        fn borrow(&self) -> &T { self }
}

trait Foo {
        fn foo(&self, other: &Self);
}

fn bar<K, Q>(k: &K, q: &Q) where K: Borrow<Q>, Q: Foo {
    q.foo(k.borrow())
}

struct MyTree<K>(K);

impl<K> MyTree<K> {
    // This caused a failure in #18906
    fn bar<Q>(k: &K, q: &Q) where K: Borrow<Q>, Q: Foo {
        q.foo(k.borrow())
    }
}

fn main() {}
