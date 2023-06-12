#![warn(clippy::suspicious_unary_op_formatting)]
#![allow(clippy::needless_if)]

#[rustfmt::skip]
fn main() {
    // weird binary operator formatting:
    let a = 42;

    if a >- 30 {}
    if a >=- 30 {}

    let b = true;
    let c = false;

    if b &&! c {}

    if a >-   30 {}

    // those are ok:
    if a >-30 {}
    if a < -30 {}
    if b && !c {}
    if a > -   30 {}
}
