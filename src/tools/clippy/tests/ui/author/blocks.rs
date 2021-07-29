#![allow(redundant_semicolons, clippy::no_effect)]

#[rustfmt::skip]
fn main() {
    #[clippy::author]
    {
        let x = 42i32;
        -x;
    };
    #[clippy::author]
    {
        let expr = String::new();
        drop(expr)
    };
}
