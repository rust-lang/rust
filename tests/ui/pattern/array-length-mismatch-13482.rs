//! Regression test for https://github.com/rust-lang/rust/issues/13482

fn main() {
  let x = [1,2];
  let y = match x {
    [] => None, //~ ERROR pattern requires 0 elements but array has 2
    [a,_] => Some(a)
  };
}
