#![feature(frontmatter)]

//@ check-pass

include!("auxiliary/lib.rs");

// auxiliary/lib.rs contains a frontmatter. Ensure that we can use them in an
// `include!` macro.

fn main() {
    foo(1);
}
