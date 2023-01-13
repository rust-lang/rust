// run-pass
#![allow(unused_variables)]
pub fn main() {
    let x = (0, 2);

    match x {
        (0, ref y) => {}
        (y, 0) => {}
        _ => (),
    }
}
