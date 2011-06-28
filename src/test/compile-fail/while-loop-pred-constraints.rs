// xfail-stage0
// error-pattern: Unsatisfied precondition constraint (for example, even(y

fn print_even(int y) : even(y) {
  log y;
}

pred even(int y) -> bool {
  true
}

fn main() {

  let int y = 42;
  let int x = 1;
  check even(y);
  while (true) {
    print_even(y);
    while (true) {
      while (true) {
	while (true) {
	  y += x;
	}
      }
    }
  }
}