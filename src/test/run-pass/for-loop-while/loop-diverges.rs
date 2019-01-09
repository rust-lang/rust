// run-pass
#![allow(unused_parens)]
// pretty-expanded FIXME #23616

/* Make sure a loop{} can be the tailexpr in the body
of a diverging function */

fn forever() -> ! {
  loop{}
}

pub fn main() {
  if (1 == 2) { forever(); }
}
