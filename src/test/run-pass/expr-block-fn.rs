// xfail-boot

fn test_fn() {
  type t = fn() -> int;
  fn ten() -> int {
    ret 10;
  }
  let t res = { ten };
  check (res() == 10);
}

fn main() {
  test_fn();
}