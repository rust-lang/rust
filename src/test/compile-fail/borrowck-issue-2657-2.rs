fn main() {
let x = some(~1);
match x {
  some(ref y) => {
    let _b <- *y; //~ ERROR moving out of dereference of immutable & pointer
  }
  _ => {}
}
}
