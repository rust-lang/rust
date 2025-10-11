// https://github.com/rust-lang/rust/issues/64792
struct X {}

const Y: X = X("ö"); //~ ERROR expected function, tuple struct or tuple variant, found struct `X`

fn main() {}
