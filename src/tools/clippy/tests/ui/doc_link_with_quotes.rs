#![warn(clippy::doc_link_with_quotes)]

fn main() {
    foo()
}

/// Calls ['bar']
pub fn foo() {
    bar()
}

pub fn bar() {}
