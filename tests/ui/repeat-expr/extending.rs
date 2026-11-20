//@ run-pass
//! Test that array repeat expressions are extending

#![allow(path_statements)]

fn temp() {}

fn main() {
    let _extended = [&temp(); 1];
    _extended;

    let _extended = [&temp(); 17];
    _extended;
}
