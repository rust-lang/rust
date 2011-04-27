// error-pattern:Unsatisfied precondition

fn foo(int x) {
  log x;
}

fn main() {
  let int x;
  if (1 > 2) {
    log "whoops";
  } else {
    x = 10;
  }
  foo(x);
}