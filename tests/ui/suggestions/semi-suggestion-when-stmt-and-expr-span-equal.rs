//! Regression test for suggestions that were fired on empty spans
//! involving macro-call statements. For some reason the semicolon
//! is not included in the overall span of the macro-call statement.
//!
//! Issue 1: <https://github.com/rust-lang/rust/issues/133833>.
//! Issue 2: <https://github.com/rust-lang/rust/issues/133834>.
//! See also: <https://github.com/rust-lang/rust/issues/133845>.

fn foo() -> String {
    let mut list = {
        println!();
    };
    list //~ ERROR mismatched types
}

fn bar() {
    String::new()
        .chars()
        .filter(|x| !x.is_whitespace())
        .map(|x| {
            println!("Child spawned with the size: {}", x);
        })
        .collect::<String>(); //~ ERROR E0277
}

fn main() {}
