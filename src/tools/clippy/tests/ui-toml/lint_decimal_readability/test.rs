#![allow(clippy::excessive_precision)]
#[deny(clippy::unreadable_literal)]

fn allow_inconsistent_digit_grouping() {
    #![allow(clippy::inconsistent_digit_grouping)]
    let _pass1 = 100_200_300.123456789;
}

fn main() {
    allow_inconsistent_digit_grouping();

    let _pass1 = 100_200_300.100_200_300;
    let _pass2 = 1.123456789;
    let _pass3 = 1.0;
    let _pass4 = 10000.00001;
    let _pass5 = 1.123456789e1;

    // due to clippy::inconsistent-digit-grouping
    let _fail1 = 100_200_300.123456789;

    // fail due to the integer part
    let _fail2 = 100200300.300200100;
}
