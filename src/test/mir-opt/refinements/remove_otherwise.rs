// compile-flags: -Z mir-opt-level=4
// EMIT_MIR remove_otherwise.foo.Refinements.diff
fn foo(a: bool) -> u32 {
    match a {
        b @ false => match b {
            true => 1,
            false => 7,
        },
        true => 28,
    }
}

fn main() {
    foo(true);
}
