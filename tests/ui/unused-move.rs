//@ run-pass
// Issue #3878
// Issue Name: Unused move causes a crash
// Abstract: zero-fill to block after drop


#![allow(path_statements)]

pub fn main() {
    let y: Box<_> = Box::new(1);
    y;
}
