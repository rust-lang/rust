/* Make sure a loop{} with a break in it can't be
   the tailexpr in the body of a diverging function */
fn forever() -> ! {
  loop {
    break;
  }
  return 42i; //~ ERROR expected `_|_` but found `int`
}

fn main() {
  if (1 == 2) { forever(); }
}