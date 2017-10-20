#![feature(plugin)]
#![plugin(clippy_mini_macro_test)]

#[deny(warnings)]
#[mini_macro_attr]
fn main() {
    let _ = mini_macro!();
}
