// run-pass
#[derive(PartialEq, Debug)]
struct Foo;

pub fn main() {
  assert_eq!(Foo, Foo);
  assert!(!(Foo != Foo));
}
