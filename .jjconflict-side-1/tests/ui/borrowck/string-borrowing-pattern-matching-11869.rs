// https://github.com/rust-lang/rust/issues/11869
//@ check-pass
#![allow(dead_code)]

struct A {
    a: String
}

fn borrow<'a>(binding: &'a A) -> &'a str {
    match &*binding.a {
        "in" => "in_",
        "ref" => "ref_",
        ident => ident
    }
}

fn main() {}
