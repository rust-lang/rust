// run-rustfix
// edition:2018

#![feature(custom_inner_attributes)]
#![feature(exclusive_range_pattern)]
#![feature(stmt_expr_attributes)]
#![warn(clippy::almost_complete_letter_range)]
#![allow(ellipsis_inclusive_range_patterns)]

macro_rules! a {
    () => {
        'a'
    };
}

fn main() {
    #[rustfmt::skip]
    {
        let _ = ('a') ..'z';
        let _ = 'A' .. ('Z');
    }

    let _ = 'b'..'z';
    let _ = 'B'..'Z';

    let _ = (b'a')..(b'z');
    let _ = b'A'..b'Z';

    let _ = b'b'..b'z';
    let _ = b'B'..b'Z';

    let _ = a!()..'z';

    let _ = match 0u8 {
        b'a'..b'z' if true => 1,
        b'A'..b'Z' if true => 2,
        b'b'..b'z' => 3,
        b'B'..b'Z' => 4,
        _ => 5,
    };

    let _ = match 'x' {
        'a'..'z' if true => 1,
        'A'..'Z' if true => 2,
        'b'..'z' => 3,
        'B'..'Z' => 4,
        _ => 5,
    };
}

fn _under_msrv() {
    #![clippy::msrv = "1.25"]
    let _ = match 'a' {
        'a'..'z' => 1,
        _ => 2,
    };
}

fn _meets_msrv() {
    #![clippy::msrv = "1.26"]
    let _ = 'a'..'z';
    let _ = match 'a' {
        'a'..'z' => 1,
        _ => 2,
    };
}
