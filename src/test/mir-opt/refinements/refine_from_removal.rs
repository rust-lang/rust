// compile-flags: -Z mir-opt-level=4
// EMIT_MIR refine_from_removal.foo.Refinements.diff
fn foo(a: u32) -> u32 {
    let c = match a {
        b @ (1 | 2) => match b {
            1 => 18,
            2 => 384,
            3 => 28732897,
            _ => 8594,
        },
        _ => 1,
    };

    match c {
        18 => 3,
        384 => 4,
        1 => 5,
        _ => 6,
        // FIXME: this currently doesnt optimise out the `1 =>` and `_ =>` arms
    }
}

fn main() {
    foo(10);
}
