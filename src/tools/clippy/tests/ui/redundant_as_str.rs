#![warn(clippy::redundant_as_str)]

fn main() {
    let string = "Hello, world!".to_owned();

    // These methods are redundant and the `as_str` can be removed
    let _redundant = string.as_str().as_bytes();
    let _redundant = string.as_str().is_empty();

    // These methods don't use `as_str` when they are redundant
    let _no_as_str = string.as_bytes();
    let _no_as_str = string.is_empty();

    // These methods are not redundant, and are equivelant to
    // doing dereferencing the string and applying the method
    let _not_redundant = string.as_str().escape_unicode();
    let _not_redundant = string.as_str().trim();
    let _not_redundant = string.as_str().split_whitespace();

    // These methods don't use `as_str` and are applied on a `str` directly
    let borrowed_str = "Hello, world!";
    let _is_str = borrowed_str.as_bytes();
    let _is_str = borrowed_str.is_empty();
}
