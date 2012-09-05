trait clam<A: copy> {
  fn chowder(y: A);
}
struct foo<A: copy> : clam<A> {
  let x: A;
  fn chowder(y: A) {
  }
}

fn foo<A: copy>(b: A) -> foo<A> {
    foo {
        x: b
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
