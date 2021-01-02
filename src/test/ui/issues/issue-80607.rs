// This tests makes sure the diagnostics print the offending enum variant, not just the type.
pub enum Enum {
    V1(i32),
}

pub fn foo(x: i32) -> Enum {
    Enum::V1 { x } //~ ERROR field does not exist
}

fn main() {}
