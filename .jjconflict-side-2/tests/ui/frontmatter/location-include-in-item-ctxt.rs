// Ensure that in item ctxts we can `include` files that contain frontmatter.
//@ check-pass

#![feature(frontmatter)]

include!("auxiliary/lib.rs");

fn main() {
    foo(1);
}
