#![warn(clippy::filter_next)]

fn main() {
    issue10029();
}

pub fn issue10029() {
    let iter = (0..10);
    let _ = iter.filter(|_| true).next();
}
