//xfail-test

// this should be illegal but borrowck is not handling 
// pattern bindings correctly right now

fn main() {
let x = some(~1);
alt x {
  some(y) {
    let b <- y;
  }
  _ {}
}
}
