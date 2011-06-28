// xfail-stage0
// error-pattern: Unsatisfied precondition constraint (for example, init(i

fn main() {
  let int i;

  log (false || {i = 5; true});
  log i;
}