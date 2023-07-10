// compile-flags: -Ztrait-solver=next
// check-pass

fn test(s: &[u8]) {
    match &s[0..3] {
        b"uwu" => {}
        _ => {}
    }
}

fn main() {}
