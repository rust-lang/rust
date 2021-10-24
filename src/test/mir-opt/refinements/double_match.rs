// compile-flags: -Z mir-opt-level=4
// EMIT_MIR double_match.foo.Refinements.diff
fn foo(a: u32) -> i32 {
    match a {
        b @ 1 => match b {
            1 => 1,
            2 => 2,
            _ => 3,
        },
        _ => 3,
    }
}

fn main() {
    foo(10);
}
