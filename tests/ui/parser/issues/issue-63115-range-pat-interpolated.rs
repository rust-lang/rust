//@ check-pass

#![allow(ellipsis_inclusive_range_patterns)]

fn main() {
    macro_rules! mac_expr {
        ($e:expr) => {
            if let 2...$e = 3 {}
            if let 2..=$e = 3 {}
            if let 2..$e = 3 {}
            if let ..$e = 3 {}
            if let ..=$e = 3 {}
            if let $e.. = 5 {}
            if let $e..5 = 4 {}
            if let $e..=5 = 4 {}
        }
    }
    mac_expr!(4);
}
