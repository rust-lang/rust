iface clam<A: copy> {
  fn chowder(y: A);
}
class foo<A: copy> : clam<A> {
  let x: A;
  new(b: A) { self.x = b; }
  fn chowder(y: A) {
  }
}

fn f<A: copy>(x: clam<A>, a: A) {
  x.chowder(a);
}

fn main() {

  let c = foo(42);
  let d: clam<int> = c as clam::<int>;
  f(d, c.x);
}
