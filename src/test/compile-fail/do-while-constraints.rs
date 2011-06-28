// xfail-stage0
// error-pattern: Unsatisfied precondition constraint (for example, init(y
fn main() {

  let int y = 42;
  let int x;
  do {
    log y;
    do {
      do {
	do {
	  x <- y;
	} while (true);
      } while (true);
    } while (true);
  } while (true);
}