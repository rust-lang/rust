// Testcase for issue #131.

fn main() -> () {
  let int a = 10;
  log a;
  check (a * (a - 1) == 90);
}
