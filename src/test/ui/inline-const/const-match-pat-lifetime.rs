// run-pass

#![allow(incomplete_features)]
#![feature(inline_const)]

// rust-lang/rust#78174: ICE: "cannot convert ReErased to a region vid"

fn main() {
    match "foo" {
        const { concat!("fo", "o") } => (),
        _ => unreachable!(),
    }
}
