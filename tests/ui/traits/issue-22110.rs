//@ run-pass
// Test an issue where we reported ambiguity between the where-clause
// and the blanket impl. The only important thing is that compilation
// succeeds here. Issue #22110.


#![allow(dead_code)]

trait Foo<A> {
    fn foo(&self, a: A);
}

impl<A,F:Fn(A)> Foo<A> for F {
    fn foo(&self, _: A) { }
}

fn baz<A,F:for<'a> Foo<(&'a A,)>>(_: F) { }

fn components<T,A>(t: fn(&A))
    where fn(&A) : for<'a> Foo<(&'a A,)>,
{
    baz(t)
}

fn main() {
}
