// Tests that we don't generate a spurious error about f.honk's type
// being undeterminable
fn main() {
  let f = 42;

  let _g = if f < 5 {
      f.honk() //~ ERROR no method named `honk` found
  }
  else {
      ()
  };
}
