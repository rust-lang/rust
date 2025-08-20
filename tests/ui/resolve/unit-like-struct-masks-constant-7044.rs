// https://github.com/rust-lang/rust/issues/7044
static X: isize = 0;
struct X; //~ ERROR the name `X` is defined multiple times

fn main() {}
