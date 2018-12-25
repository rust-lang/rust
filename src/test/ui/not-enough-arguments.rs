// Check that the only error msg we report is the
// mismatch between the # of params, and not other
// unrelated errors.

fn foo(a: isize, b: isize, c: isize, d:isize) {
  panic!();
}

fn main() {
  foo(1, 2, 3);
  //~^ ERROR this function takes 4 parameters but 3
}
