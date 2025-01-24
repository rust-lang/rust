//@ run-pass
#![allow(dead_code)]
// Issue #961


fn altsimple() {
    match Box::new(true) {
      _ => { }
    }
}
pub fn main() { }
