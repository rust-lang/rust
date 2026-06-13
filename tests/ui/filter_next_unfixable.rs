//@no-rustfix
#![warn(clippy::filter_next)]

#[rustfmt::skip]
fn main() {
    let v = [3, 2, 1, 0, -1, -2, -3];

    // Multi-line case -- only a note is emitted
    let _ = v.iter().filter(|&x| {
    //~^ filter_next
                                *x < 0
                            }
                   ).next();

    let _ = v.iter().filter(|&x| {
    //~^ filter_next
                                *x < 0
                            }
                   ).next_back();
}

// The fixed version doesn't compile, as `iter` isn't `mut`.
// We do emit a note suggesting adding it, but not an autofix
pub fn issue10029() {
    {
        let iter = (0..10);
        let _ = iter.filter(|_| true).next();
        //~^ filter_next
    }
    {
        let iter = (0..10);
        let _ = iter.filter(|_| true).next_back();
        //~^ filter_next
    }
}
