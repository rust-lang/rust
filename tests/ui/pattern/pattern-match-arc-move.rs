//! Tests moving an `Arc` value out of an `Option` in a match expression.

//@ run-pass

use std::sync::Arc;
fn dispose(_x: Arc<bool>) { }

pub fn main() {
    let p = Arc::new(true);
    let x = Some(p);
    match x {
        Some(z) => { dispose(z); },
        None => panic!()
    }
}
