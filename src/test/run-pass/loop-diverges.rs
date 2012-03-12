/* Make sure a loop{} can be the tailexpr in the body
of a diverging function */

fn forever() -> ! {
  loop{}
}

fn main() {
  if (1 == 2) { forever(); }
}