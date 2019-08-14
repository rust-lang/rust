// run-pass
struct Foo<A: Repr>(<A as Repr>::Data);

impl<A> Copy for Foo<A> where <A as Repr>::Data: Copy { }
impl<A> Clone for Foo<A> where <A as Repr>::Data: Clone {
    fn clone(&self) -> Self { Foo(self.0.clone()) }
}

trait Repr {
    type Data;
}

impl<A> Repr for A {
    type Data = u32;
}

fn main() {
}
