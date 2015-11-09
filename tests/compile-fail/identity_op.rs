#![feature(plugin)]
#![plugin(clippy)]

const ONE : i64 = 1;
const NEG_ONE : i64 = -1;
const ZERO : i64 = 0;

#[deny(identity_op)]
fn main() {
    let x = 0;

    x + 0;        //~ERROR the operation is ineffective
    x + (1 - 1);  //~ERROR the operation is ineffective
    x + 1;
    0 + x;        //~ERROR the operation is ineffective
    1 + x;
    x - ZERO;     //no error, as we skip lookups (for now)
    x | (0);      //~ERROR the operation is ineffective
    ((ZERO)) | x; //no error, as we skip lookups (for now)

    x * 1;        //~ERROR the operation is ineffective
    1 * x;        //~ERROR the operation is ineffective
    x / ONE;      //no error, as we skip lookups (for now)

    x / 2;        //no false positive

    x & NEG_ONE;  //no error, as we skip lookups (for now)
    -1 & x;       //~ERROR the operation is ineffective
}
