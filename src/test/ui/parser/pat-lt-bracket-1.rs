fn main() {
  match 42 {
    x < 7 => (),
   //~^ ERROR expected one of
    _ => ()
  }
}
