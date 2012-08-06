fn main() {
let x = some(~1);
match x {
  some(y) => {
    let _b <- y; //~ ERROR moving out of pattern binding
  }
  _ => {}
}
}
