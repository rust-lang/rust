// run-rustfix

#![allow(
    dead_code,
    unused_variables,
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping
)]

fn main() {
    let fail14 = 2_32;
    let fail15 = 4_64;
    let fail16 = 7_8; //
    let fail17 = 23_16; //
    let ok18 = 23_128;

    let fail20 = 2__8; //
    let fail21 = 4___16; //

    let ok24 = 12.34_64;
    let fail25 = 1E2_32;
    let fail26 = 43E7_64;
    let fail27 = 243E17_32;
    #[allow(overflowing_literals)]
    let fail28 = 241251235E723_64;
    let ok29 = 42279.911_32;

    let _ = 1.12345E1_32;
}
