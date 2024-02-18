//@ check-pass
//@ edition:2021
#![deny(rust_2021_compatibility)]

pub struct Warns {
    // `Arc` has significant drop
    _significant_drop: std::sync::Arc<()>,
    field: String,
}

pub fn test(w: Warns) {
    _ = || drop(w.field);
}

fn main() {}
