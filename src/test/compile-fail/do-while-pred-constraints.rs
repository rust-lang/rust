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
  check even(y);
  do {
    print_even(y);
    do {
      do {
	do {
	  y += 1;
	} while (true);
      } while (true);
    } while (true);
  } while (true);
}