// build-pass

#![allow(incomplete_features)]
#![feature(inline_const)]
fn main() {
    const N: u32 = 10;
    let x: u32 = 3;

    match x {
        const { N - 1 } ..= 10 => {},
        _ => {},
    }

    match x {
        const { N - 1 } ..= const { N + 1 } => {},
        _ => {},
    }

    match x {
        1 ..= const { N + 1 } => {},
        _ => {},
    }
}
