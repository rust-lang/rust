//@ check-pass
// Ensure that a proper body is used when printing paths (`x`)
// if the attribute is placed on an item.
#[clippy::author]
fn main() {
    let x = 42i32;
    _ = -x;
}
