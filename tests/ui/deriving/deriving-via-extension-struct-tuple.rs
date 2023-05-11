// run-pass
#[derive(PartialEq, Debug)]
struct Foo(isize, isize, String);

pub fn main() {
  let a1 = Foo(5, 6, "abc".to_string());
  let a2 = Foo(5, 6, "abc".to_string());
  let b = Foo(5, 7, "def".to_string());

  assert_eq!(a1, a1);
  assert_eq!(a2, a1);
  assert!(!(a1 == b));

  assert!(a1 != b);
  assert!(!(a1 != a1));
  assert!(!(a2 != a1));
}
