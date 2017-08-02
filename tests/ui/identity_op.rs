#![feature(plugin)]
#![plugin(clippy)]

const ONE : i64 = 1;
const NEG_ONE : i64 = -1;
const ZERO : i64 = 0;

#[allow(eq_op, no_effect, unnecessary_operation, double_parens)]
#[warn(identity_op)]
fn main() {
    let x = 0;

    x + 0;
    x + (1 - 1);
    x + 1;
    0 + x;
    1 + x;
    x - ZERO;     //no error, as we skip lookups (for now)
    x | (0);
    ((ZERO)) | x; //no error, as we skip lookups (for now)

    x * 1;
    1 * x;
    x / ONE;      //no error, as we skip lookups (for now)

    x / 2;        //no false positive

    x & NEG_ONE;  //no error, as we skip lookups (for now)
    -1 & x;
}
