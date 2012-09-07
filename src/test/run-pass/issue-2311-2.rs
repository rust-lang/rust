trait clam<A: Copy> { }
struct foo<A: Copy> {
    x: A,
   fn bar<B,C:clam<A>>(c: C) -> B {
     fail;
   }
}

fn foo<A: Copy>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

fn main() { }
