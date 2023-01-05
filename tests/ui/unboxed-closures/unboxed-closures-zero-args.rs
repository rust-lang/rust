// run-pass
#![allow(unused_mut)]
// pretty-expanded FIXME #23616

fn main() {
    let mut zero = || {};
    let () = zero();
}
