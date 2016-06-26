#![feature(plugin)]
#![plugin(clippy, clippy_mini_macro_test)]

#[deny(warnings)]
fn main() {
    let _ = mini_macro!();
}
