// compile-flags: -Z mir-opt-level=4
// EMIT_MIR otherwise_refinements.foo.Refinements.diff
fn foo(a: bool) -> u32 {
    match a {
        b @ true => match b {
            true => 1,
            false => 7,
        },
        false => 28,
    }
}

fn main() {
    foo(true);
}
