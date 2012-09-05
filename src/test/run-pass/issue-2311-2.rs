trait clam<A: copy> { }
struct foo<A: copy> {
  let x: A;
   fn bar<B,C:clam<A>>(c: C) -> B {
     fail;
   }
}

fn foo<A: copy>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

fn main() { }
