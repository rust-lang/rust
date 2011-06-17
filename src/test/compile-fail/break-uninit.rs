// error-pattern:Unsatisfied precondition

fn foo() -> int {
  let int x;
  let int i;

  do {
    i = 0;
    break;
    x = 0;
  } while (x != 0);

  log(x);

  ret 17;
}

fn main() {
  log(foo());
}
