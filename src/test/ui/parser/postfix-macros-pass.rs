// check-pass

// Basically a clone of postfix-macros.rs, but with the offending
// code behind a `#[cfg(FALSE)]`. Rust still parses this code,
// but doesn't do anything beyond with it.

fn main() {}

#[cfg(FALSE)]
fn foo() {
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
