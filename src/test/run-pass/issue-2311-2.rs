iface clam<A: copy> { }
class foo<A: copy> {
  let x: A;
  new(b: A) { self.x = b; }
   fn bar<B,C:clam<A>>(c: C) -> B {
     fail;
   }
}

fn main() { }
