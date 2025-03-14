//@ compile-flags: -Zlint-mir
//@ check-pass

#![feature(inline_const_pat)]

fn main() {
    match 1 {
        const {
            || match 0 {
                x => 0,
            };
            0
        } => (),
        _ => (),
    }
}
