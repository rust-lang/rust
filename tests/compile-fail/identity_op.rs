#![feature(plugin)]
#![plugin(clippy)]

const ONE : i64 = 1;
const NEG_ONE : i64 = -1;
const ZERO : i64 = 0;

#[deny(identity_op)]
fn main() {
    let x = 0;

    x + 0; //~ERROR
    0 + x; //~ERROR
    x - ZERO; //~ERROR
    x | (0); //~ERROR
    ((ZERO)) | x; //~ERROR

    x * 1; //~ERROR
    1 * x; //~ERROR
    x / ONE; //~ERROR

    x & NEG_ONE; //~ERROR
    -1 & x; //~ERROR
}
