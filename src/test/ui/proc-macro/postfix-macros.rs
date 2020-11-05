// run-pass
// aux-build:demacroify.rs

extern crate demacroify;

#[demacroify::demacroify]
fn main() {
    "Hello, world!".to_string().println!();

    "Hello, world!".println!();

    false.assert!();

    Some(42).assert_eq!(None);

    std::iter::once(42)
        .map(|v| v + 3)
        .dbg!()
        .max()
        .unwrap()
        .dbg!();
}
