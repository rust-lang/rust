fn main() {
let x = Some(~1);
match x { //~ NOTE loan of immutable local variable granted here
  Some(ref _y) => {
    let _a = x; //~ ERROR moving out of immutable local variable prohibited due to outstanding loan
  }
  _ => {}
}
}
