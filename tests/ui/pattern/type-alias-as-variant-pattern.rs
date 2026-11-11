//! Regression test for <https://github.com/rust-lang/rust/issues/42880>.
//! Test type-based paths in variant patterns don't cause ICE.

type Value = String;

fn main() {
    let f = |&Value::String(_)| (); //~ ERROR no associated function or constant named

    let vec: Vec<Value> = Vec::new();
    vec.last().map(f);
}
