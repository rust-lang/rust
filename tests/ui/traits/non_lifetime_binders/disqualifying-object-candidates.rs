//@ check-pass

trait Foo {
    type Bar<T>
    where
        dyn Send + 'static: Send;
}

impl Foo for () {
    type Bar<T> = i32;
    // We take `<() as Foo>::Bar<T>: Sized` and normalize it under the where clause
    // of `for<S> <() as Foo>::Bar<S> = i32`. This gives us back `i32: Send` with
    // the nested obligation `(dyn Send + 'static): Send`. However, during candidate
    // assembly for object types, we disqualify any obligations that has non-region
    // late-bound vars in the param env(!), rather than just the predicate. This causes
    // the where clause to not hold even though it trivially should.
}

fn main() {}
