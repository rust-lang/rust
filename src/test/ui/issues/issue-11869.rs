// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
