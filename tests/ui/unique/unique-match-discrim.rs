// run-pass
#![allow(dead_code)]
// Issue #961

// pretty-expanded FIXME #23616

fn altsimple() {
    match Box::new(true) {
      _ => { }
    }
}
pub fn main() { }
