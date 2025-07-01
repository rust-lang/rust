//! Check that a `mismatched types` error (E0308) is correctly reported when attempting to
//! bind a reference to an `i32` to a reference to a `String`.
//! This specifically tests type checking for `ref` bindings.

fn main() {
    let var = 10i32;
    let ref string: String = var; //~ ERROR mismatched types [E0308]
}
