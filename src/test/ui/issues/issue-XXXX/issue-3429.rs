// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
  let x = 1_usize;
  let y = || x;
  let _z = y();
}
