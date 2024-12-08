#![warn(clippy::obfuscated_if_else)]

fn main() {
    true.then_some("a").unwrap_or("b");
}
