#![feature(slice_patterns)]

enum t { a(u), b }
enum u { c, d }

fn match_nested_vecs<'a, T>(l1: Option<&'a [T]>, l2: Result<&'a [T], ()>) -> &'static str {
    match (l1, l2) { //~ ERROR non-exhaustive patterns: `(Some(&[]), Err(_))` not covered
        (Some(&[]), Ok(&[])) => "Some(empty), Ok(empty)",
        (Some(&[_, ..]), Ok(_)) | (Some(&[_, ..]), Err(())) => "Some(non-empty), any",
        (None, Ok(&[])) | (None, Err(())) | (None, Ok(&[_])) => "None, Ok(less than one element)",
        (None, Ok(&[_, _, ..])) => "None, Ok(at least two elements)"
    }
}

fn main() {
    let x = t::a(u::c);
    match x { //~ ERROR non-exhaustive patterns: `a(c)` not covered
        t::a(u::d) => { panic!("hello"); }
        t::b => { panic!("goodbye"); }
    }
}
