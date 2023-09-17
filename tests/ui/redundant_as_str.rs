#![warn(clippy::redundant_as_str)]

fn main() {
    let _redundant = "Hello, world!".to_owned().as_str().as_bytes();
    let _redundant = "Hello, world!".to_owned().as_str().is_empty();
}
