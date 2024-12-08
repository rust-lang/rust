//@ run-pass

pub fn main() {
  let x = 1_usize;
  let y = || x;
  let _z = y();
}
