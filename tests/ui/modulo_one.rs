#![warn(clippy::modulo_one)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::identity_op)]

static STATIC_ONE: usize = 2 - 1;
static STATIC_NEG_ONE: i64 = 1 - 2;

fn main() {
    10 % 1;
    10 % -1;
    10 % 2;
    // also caught by rustc
    i32::MIN % (-1);

    const ONE: u32 = 1 * 1;
    const NEG_ONE: i64 = 1 - 2;
    const INT_MIN: i64 = i64::MIN;

    2 % ONE;
    // NOT caught by lint
    5 % STATIC_ONE;
    2 % NEG_ONE;
    // NOT caught by lint
    5 % STATIC_NEG_ONE;
    // also caught by rustc
    INT_MIN % NEG_ONE;
    // ONLY caught by rustc
    INT_MIN % STATIC_NEG_ONE;
}
