// build-pass

#![feature(generators)]

#![allow(unused_assignments, dead_code)]

fn main() {
    let _ = || {
        let mut x = vec![22_usize];
        std::mem::drop(x);
        match y() {
            true if {
                x = vec![];
                false
            } => {}
            _ => {
                yield;
            }
        }
    };
}

fn y() -> bool {
    true
}
