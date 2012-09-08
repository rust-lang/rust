trait clam<A: Copy> {
  fn chowder(y: A);
}
struct foo<A: Copy> {
  x: A,
}

impl<A: Copy> foo<A> : clam<A> {
  fn chowder(y: A) {
  }
}

fn foo<A: Copy>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

fn f<A: Copy>(x: clam<A>, a: A) {
  x.chowder(a);
}

fn main() {

  let c = foo(42);
  let d: clam<int> = c as clam::<int>;
  f(d, c.x);
}
