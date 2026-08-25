// https://github.com/rust-lang/rust/issues/80607
// This tests makes sure the diagnostics print the offending enum variant, not just the type.
pub enum Enum {
    V1(i32),
}

pub fn foo(x: i32) -> Enum {
    Enum::V1 { x } //~ ERROR `Enum::V1` has no field named `x`
}

fn main() {}
