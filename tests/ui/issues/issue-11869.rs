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
