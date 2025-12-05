// https://github.com/rust-lang/rust/issues/84434
//@ check-pass

use std::path::Path;
struct A {
    pub func: fn(check: bool, a: &Path, b: Option<&Path>),
}
const MY_A: A = A {
    func: |check, a, b| {
        if check {
            let _ = ();
        } else if let Some(parent) = b.and_then(|p| p.parent()) {
            let _ = ();
        }
    },
};

fn main() {}
