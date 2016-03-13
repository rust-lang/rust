#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn return_none() -> Option<i64> {
    None
}

#[miri_run]
fn return_some() -> Option<i64> {
    Some(42)
}

// #[miri_run]
// fn match_opt_none() -> i64 {
//     let x = None,
//     match x {
//         Some(data) => data,
//         None => 42,
//     }
// }

// #[miri_run]
// fn match_opt_some() -> i64 {
//     let x = Some(13);
//     match x {
//         Some(data) => data,
//         None => 42,
//     }
// }
