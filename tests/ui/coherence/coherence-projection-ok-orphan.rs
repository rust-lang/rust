// Here we do not get a coherence conflict because `Baz: Iterator`
// does not hold and (due to the orphan rules), we can rely on that.

//@ check-pass

pub trait Foo<P> {}

pub trait Bar {
    type Output: 'static;
}

struct Baz;
impl Foo<i32> for Baz { }

impl<A:Iterator> Foo<A::Item> for A { }

fn main() {}
