#![allow(dead_code, unused_variables)]

/// Should not trigger an ICE in `SpanlessHash` / `consts::constant`
///
/// Issue: https://github.com/rust-lang/rust-clippy/issues/2594

fn spanless_hash_ice() {
    let txt = "something";
    let empty_header: [u8; 1] = [1; 1];

    match txt {
        "something" => {
            let mut headers = [empty_header; 1];
        },
        "" => (),
        _ => (),
    }
}

fn main() {}
