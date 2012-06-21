fn main() {
let x = some(~1);
alt x {
  some(y) {
    let _b <- y; //! ERROR moving out of pattern binding
  }
  _ {}
}
}
