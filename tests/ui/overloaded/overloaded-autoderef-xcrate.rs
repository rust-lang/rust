// run-pass
// aux-build:overloaded_autoderef_xc.rs


extern crate overloaded_autoderef_xc;

fn main() {
    assert!(overloaded_autoderef_xc::check(5, 5));
}
