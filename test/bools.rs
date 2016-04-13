#![crate_type = "lib"]
#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn boolean() -> bool {
    true
}

#[miri_run]
fn if_false() -> i64 {
    let c = false;
    if c { 1 } else { 0 }
}

#[miri_run]
fn if_true() -> i64 {
    let c = true;
    if c { 1 } else { 0 }
}

#[miri_run]
fn match_bool() -> i16 {
    let b = true;
    match b {
        true => 1,
        _ => 0,
    }
}
