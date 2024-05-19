// Check that storage statements reset local qualification.
//@ check-pass
use std::cell::Cell;

const C: Option<Cell<u32>> = {
    let mut c = None;
    let mut i = 0;
    while i == 0 {
        let mut x = None;
        c = x;
        x = Some(Cell::new(0));
        let _ = x;
        i += 1;
    }
    c
};

fn main() {
    let _: &'static _ = &C;
}
