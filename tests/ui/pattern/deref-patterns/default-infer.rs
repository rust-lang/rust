//@ check-pass
#![feature(deref_patterns)]
#![expect(incomplete_features)]

fn main() {
    match <_ as Default>::default() {
        "" => (),
        _ => unreachable!(),
    }
}
