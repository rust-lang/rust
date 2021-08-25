// run-pass
// Issue #3878
// Issue Name: Unused move causes a crash
// Abstract: zero-fill to block after drop

// pretty-expanded FIXME #23616

#![allow(path_statements)]

pub fn main() {
    let y: Box<_> = Box::new(1);
    y;
}
