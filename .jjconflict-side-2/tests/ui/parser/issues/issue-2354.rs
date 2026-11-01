fn foo() { //~ NOTE unclosed delimiter
  match Some(10) {
  //~^ NOTE this delimiter might not be properly closed...
      Some(y) => { panic!(); }
      None => { panic!(); }
}
//~^ NOTE ...as it matches this but it has different indentation

fn bar() {
    let mut i = 0;
    while (i < 1000) {}
}

fn main() {}
//~ ERROR this file contains an unclosed delimiter
