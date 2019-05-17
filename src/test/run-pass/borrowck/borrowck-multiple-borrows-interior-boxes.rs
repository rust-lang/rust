// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test case from #39963.

#[derive(Clone)]
struct Foo(Option<Box<Foo>>, Option<Box<Foo>>);

fn test(f: &mut Foo) {
  match *f {
    Foo(Some(ref mut left), Some(ref mut right)) => match **left {
      Foo(Some(ref mut left), Some(ref mut right)) => panic!(),
      _ => panic!(),
    },
    _ => panic!(),
  }
}

fn main() {
}
