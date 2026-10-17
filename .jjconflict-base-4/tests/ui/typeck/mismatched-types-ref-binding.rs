//! Check that a `mismatched types` error (E0308) is correctly reported when attempting to
//! bind a reference to an `i32` to a reference to a `String`.
//! Ensure `ref` bindings report a mismatched type error.

fn main() {
    let var = 10i32;
    let ref string: String = var; //~ ERROR mismatched types [E0308]
}
