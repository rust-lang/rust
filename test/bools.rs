#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn boolean() -> bool {
    true
}

#[miri_run]
fn if_false() -> i64 {
    if false { 1 } else { 0 }
}

#[miri_run]
fn if_true() -> i64 {
    if true { 1 } else { 0 }
}

// #[miri_run]
// fn match_bool() -> i64 {
//     let b = true;
//     match b {
//         true => 1,
//         false => 0,
//     }
// }
