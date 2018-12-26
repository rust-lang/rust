#![allow(unused_must_use)]

pub fn main() {
    // Test that lambdas behave as unary expressions with block-like expressions
    -if true { 1 } else { 2 } * 3;
    || if true { 1 } else { 2 } * 3;

    // The following is invalid and parses as `if true { 1 } else { 2 }; *3`
    // if true { 1 } else { 2 } * 3
}
