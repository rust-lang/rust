// run-pass
pub fn foo(x: i8) -> i32 {
  match x {
    1 => 0,
    _ => 1,
  }
}

fn main() {
  assert_eq!(foo(0), 1);
  assert_eq!(foo(1), 0);
}
