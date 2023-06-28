// compile-flags: -Ztrait-solver=next
// known-bug: rust-lang/trait-system-refactor-initiative#38

fn test(s: &[u8]) {
    match &s[0..3] {
        b"uwu" => {}
        _ => {}
    }
}

fn main() {}
