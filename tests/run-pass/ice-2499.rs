#![allow(dead_code)]

/// Should not trigger an ICE in `SpanlessHash` / `consts::constant`
///
/// Issue: https://github.com/rust-lang-nursery/rust-clippy/issues/2499

fn f(s: &[u8]) -> bool {
    let t = s[0] as char;

    match t {
        'E' | 'W' => {}
        'T' => if &s[0..(0 + 4)] != &['0' as u8; 4] {
            return false;
        } else {
            return true;
        },
        _ => {
            return false;
        }
    }
    true
}

fn main() {}
