// The test confirms ICE-138361 is fixed.
fn main() {
  [0; loop{}]; //~ ERROR constant evaluation is taking a long time
  std::mem::transmute(4)
}
