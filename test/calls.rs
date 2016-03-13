#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// #[miri_run(expected = "Int(2)")]
// fn call() -> i32 {
//     fn increment(x: i32) -> i32 {
//         x + 1
//     }

//     increment(1)
// }

// #[miri_run(expected = "Int(3628800)")]
// fn factorial_recursive() -> i32 {
//     fn fact(n: i32) -> i32 {
//         if n == 0 {
//             1
//         } else {
//             n * fact(n - 1)
//         }
//     }

//     fact(10)
// }

// Test calling a very simple function from the standard library.
// #[miri_run(expected = "Int(1)")]
// fn cross_crate_fn_call() -> i32 {
//     if 1i32.is_positive() { 1 } else { 0 }
// }
