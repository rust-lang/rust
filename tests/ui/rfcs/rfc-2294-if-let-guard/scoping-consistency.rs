// Check that temporaries in if-let guards are correctly scoped.

//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@ compile-flags: -Zvalidate-mir

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
