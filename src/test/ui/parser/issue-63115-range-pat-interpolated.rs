// check-pass

#![feature(exclusive_range_pattern)]

#![allow(ellipsis_inclusive_range_patterns)]

fn main() {
    macro_rules! mac_expr {
        ($e:expr) => {
            if let 2...$e = 3 {}
            if let 2..=$e = 3 {}
            if let 2..$e = 3 {}
        }
    }
    mac_expr!(4);
}
