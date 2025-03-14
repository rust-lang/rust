// Tests for the issue in #137589


macro_rules! foo {
    ($x:expr) => {"rlib"}
}

#[crate_type = foo!()] //~ ERROR unexpected end of macro invocation
fn main() {}
