// error-pattern:some control paths may return
/* Make sure a loop{} with a break in it can't be
   the tailexpr in the body of a diverging function */
fn forever() -> ! {
  loop {
    break;
  }
  ret 42;
}

fn main() {
  if (1 == 2) { forever(); }
}