// Macros can be used for (parts of) the pattern and expression in an if let guard
//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

#![feature(let_chains)]

macro_rules! m {
    (pattern $i:ident) => { Some($i) };
    (expression $e:expr) => { $e };
}

fn main() {
    match () {
        () if let m!(pattern x) = m!(expression Some(4)) => {}
        () if let [m!(pattern y)] = [Some(8 + m!(expression 4))] => {}
        _ => {}
    }
}
