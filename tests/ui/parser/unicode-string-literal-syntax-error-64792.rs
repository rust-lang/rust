// https://github.com/rust-lang/rust/issues/64792
struct X {}

const Y: X = X("ö"); //~ ERROR cannot find function, tuple struct or tuple variant `X` in this scope

fn main() {}
