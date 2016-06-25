#![feature(plugin)]
#![plugin(clippy, mini_macro)]

#[deny(warnings)]
fn main() {
    let _ = mini_macro!();
}
