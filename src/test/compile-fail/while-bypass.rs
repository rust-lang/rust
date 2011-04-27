// error-pattern: precondition constraint

fn f() -> int {
  let int x;
  while(true) {
    x = 10;
  }
  ret x;
}

fn main() {
  f();
}
