//@ run-pass

use std::mem::swap;

pub fn main() {
    let mut x = 4;

    for i in 0_usize..3 {
        // ensure that the borrow in this alt
        // does not interfere with the swap
        // below.  note that it would it you
        // naively borrowed &x for the lifetime
        // of the variable x, as we once did
        match i {
            i => {
                let y = &x;
                assert!(i < *y);
            }
        }
        let mut y = 4;
        swap(&mut y, &mut x);
    }
}
