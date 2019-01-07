fn foo() { //~ NOTE un-closed delimiter
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

fn main() {} //~ NOTE here is a function named 'main'
             //~ ERROR this file contains an un-closed delimiter
