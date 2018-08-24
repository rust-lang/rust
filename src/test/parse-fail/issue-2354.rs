// compile-flags: -Z parse-only

fn foo() { //~ HELP did you mean to close this delimiter?
  match Some(x) {
      Some(y) => { panic!(); }
      None => { panic!(); }
}

fn bar() {
    let mut i = 0;
    while (i < 1000) {}
}

fn main() {} //~ ERROR this file contains an un-closed delimiter
