#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// enum MyOption<T> {
//     Some { data: T },
//     None,
// }

// #[miri_run(expected = "Int(13)")]
// fn match_my_opt_some() -> i32 {
//     let x = MyOption::Some { data: 13 };
//     match x {
//         MyOption::Some { data } => data,
//         MyOption::None => 42,
//     }
// }

// #[miri_run(expected = "Int(42)")]
// fn match_my_opt_none() -> i32 {
//     let x = MyOption::None;
//     match x {
//         MyOption::Some { data } => data,
//         MyOption::None => 42,
//     }
// }

// #[miri_run(expected = "Int(13)")]
// fn match_opt_some() -> i32 {
//     let x = Some(13);
//     match x {
//         Some(data)  => data,
//         None => 42,
//     }
// }
