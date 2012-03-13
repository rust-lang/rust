// error-pattern:non-call expression in tail call

fn f() -> int {
  let x = 1;
  be x;
}

fn main() {
  let y = f();
}
