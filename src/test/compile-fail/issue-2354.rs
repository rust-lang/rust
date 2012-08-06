// xfail-test
/*
  Ideally, the error about the missing close brace in foo would be reported
  near the corresponding open brace. But currently it's reported at the end.
  xfailed for now (see Issue #2354)
 */
fn foo() { //~ ERROR this open brace is not closed
  match some(x) {
      some(y) { fail; }
      none    { fail; }
}

fn bar() {
    let mut i = 0;
    while (i < 1000) {}
}

fn main() {}
