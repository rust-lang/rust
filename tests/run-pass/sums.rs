#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[derive(Debug, PartialEq)]
enum Unit { Unit }

#[miri_run]
fn return_unit() -> Unit {
    Unit::Unit
}

#[derive(Debug, PartialEq)]
enum MyBool { False, True }

#[miri_run]
fn return_true() -> MyBool {
    MyBool::True
}

#[miri_run]
fn return_false() -> MyBool {
    MyBool::False
}

#[miri_run]
fn return_none() -> Option<i64> {
    None
}

#[miri_run]
fn return_some() -> Option<i64> {
    Some(42)
}

#[miri_run]
fn match_opt_none() -> i8 {
    let x = None;
    match x {
        Some(data) => data,
        None => 42,
    }
}

#[miri_run]
fn match_opt_some() -> i8 {
    let x = Some(13);
    match x {
        Some(data) => data,
        None => 20,
    }
}

#[miri_run]
fn two_nones() -> (Option<i16>, Option<i16>) {
    (None, None)
}

#[miri_run]
fn main() {
    //assert_eq!(two_nones(), (None, None));
    assert_eq!(match_opt_some(), 13);
    assert_eq!(match_opt_none(), 42);
    //assert_eq!(return_some(), Some(42));
    //assert_eq!(return_none(), None);
    //assert_eq!(return_false(), MyBool::False);
    //assert_eq!(return_true(), MyBool::True);
    //assert_eq!(return_unit(), Unit::Unit);
}
