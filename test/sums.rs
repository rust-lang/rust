#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

enum Unit { Unit }

#[miri_run]
fn return_unit() -> Unit {
    Unit::Unit
}

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
    let x = None::<i32>;
    match x {
        Some(_) => 10,
        None => 20,
    }
}

#[miri_run]
fn match_opt_some() -> i8 {
    let x = Some(13);
    match x {
        Some(_) => 10,
        None => 20,
    }
}
