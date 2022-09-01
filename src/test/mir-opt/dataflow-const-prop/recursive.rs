// unit-test: DataflowConstProp

enum S<'a> {
    Rec(&'a S<'a>),
    Num(u32),
}

// EMIT_MIR recursive.main.DataflowConstProp.diff
fn main() {
    // FIXME: This currently does not work, because downcasts are rejected.
    let a = S::Num(0);
    let b = S::Rec(&a);
    let c = S::Rec(&b);
    let d = match c {
        S::Rec(b) => match b {
            S::Rec(a) => match a {
                S::Num(num) => *num,
                _ => std::process::exit(0),
            },
            _ => std::process::exit(0),
        },
        _ => std::process::exit(0),
    };
}
