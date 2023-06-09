// Regression test for #108132: do not ICE upon unmet trait alias constraint in generic impl

#![feature(trait_alias)]

trait IteratorAlias = Iterator;

struct Foo<I>(I);

impl<I: IteratorAlias> Foo<I> {
    fn f() {}
}

fn main() {
    Foo::<()>::f() //~ trait bounds were not satisfied
}
