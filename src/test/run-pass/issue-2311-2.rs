trait clam<A> {
    fn get(self) -> A;
}

struct foo<A> {
    x: A,
}

impl<A> foo<A> {
   pub fn bar<B,C:clam<A>>(&self, _c: C) -> B {
     panic!();
   }
}

fn foo<A>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

pub fn main() { }
