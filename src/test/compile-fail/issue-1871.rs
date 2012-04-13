// xfail-test
fn main() {
  let f = 42;

  let _g = if f < 5 {
      f.honk();
  }
  else {
    12
  };
}
