//@no-rustfix
#![warn(clippy::filter_next)]

fn main() {}

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
