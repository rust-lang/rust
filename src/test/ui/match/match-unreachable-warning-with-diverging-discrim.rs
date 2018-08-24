#![allow(unused_parens)]
#![deny(unreachable_code)]

fn main() {
    match (return) { } //~ ERROR unreachable expression
}
