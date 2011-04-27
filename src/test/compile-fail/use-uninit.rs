// error-pattern:Unsatisfied precondition

fn foo(int x) {
  log x;
}

fn main() {
  let int x;
  foo(x);
}