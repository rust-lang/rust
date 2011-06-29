// xfail-stage0
// error-pattern: Unsatisfied precondition constraint (for example, init(y
fn main() {

  let int y = 42;
  let int x;
  while (true) {
    log y;
    while (true) {
      while (true) {
    while (true) {
      x <- y;
    }
      }
    }
  }
}