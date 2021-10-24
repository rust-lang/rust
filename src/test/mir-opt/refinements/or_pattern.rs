// compile-flags: -Z mir-opt-level=4
// EMIT_MIR or_pattern.foo.Refinements.diff
fn foo(b: u32) -> u32 {
    match b {
        c @ (1 | 2) => match c {
            1 => 18,
            2 => 384,
            3 => 28732897,
            _ => 8594,
        },
        _ => 1,
    }
}

fn main() {
    foo(2);
}
