//@ check-pass
#![feature(string_deref_patterns)]

fn main() {
    match <_ as Default>::default() {
        "" => (),
        _ => unreachable!(),
    }
}
