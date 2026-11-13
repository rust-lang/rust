//@ check-pass
//@ edition: 2021

//@ proc-macro: print-tokens.rs
extern crate print_tokens;

fn main() {
    print_tokens::print_tokens! {
        1
        17u8
        42.
        3.14f32
        b'a'
        b'\xFF'
        'c'
        '\x32'
        "\"str\""
        r#""raw" str"#
        r###"very ##"raw"## str"###
        b"\"byte\" str"
        br#""raw" "byte" str"#
        c"\"c\" str"
        cr#""raw" "c" str"#
    }
}
