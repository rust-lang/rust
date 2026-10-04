struct S {
    f: u8,
}

fn main() {
    match (S { f: 42 }) {
        S { f: Ok(_) } => {} //~ ERROR mismatched types
    }
}
