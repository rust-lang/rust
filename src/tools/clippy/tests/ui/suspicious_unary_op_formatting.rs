#![warn(clippy::suspicious_unary_op_formatting)]
#![allow(clippy::needless_if)]

#[rustfmt::skip]
fn main() {
    // weird binary operator formatting:
    let a = 42;

    if a >- 30 {}
    //~^ ERROR: by not having a space between `>` and `-` it looks like `>-` is a single o
    if a >=- 30 {}
    //~^ ERROR: by not having a space between `>=` and `-` it looks like `>=-` is a single

    let b = true;
    let c = false;

    if b &&! c {}
    //~^ ERROR: by not having a space between `&&` and `!` it looks like `&&!` is a single

    if a >-   30 {}
    //~^ ERROR: by not having a space between `>` and `-` it looks like `>-` is a single o

    // those are ok:
    if a >-30 {}
    if a < -30 {}
    if b && !c {}
    if a > -   30 {}
}
