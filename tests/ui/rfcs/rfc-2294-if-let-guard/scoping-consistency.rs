// Check that temporaries in if-let guards are correctly scoped.

//@ build-pass
// -Zvalidate-mir

#![feature(if_let_guard)]

fn fun() {
    match 0 {
        _ => (),
        _ if let Some(s) = std::convert::identity(&Some(String::new())) => {}
        _ => (),
    }
}

fn funner() {
    match 0 {
        _ => (),
        _ | _ if let Some(s) = std::convert::identity(&Some(String::new())) => {}
        _ => (),
    }
}

fn main() {}
