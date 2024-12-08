#![warn(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::unnecessary_literal_unwrap)]
//@no-rustfix
struct Deep(Option<usize>);

#[derive(Copy, Clone)]
struct SomeStruct {
    some_field: usize,
}

fn main() {
    // fix will break type inference
    let _ = Ok(1).unwrap_or_else(|()| 2);
    //~^ ERROR: unnecessary closure used to substitute value for `Result::Err`
    //~| NOTE: `-D clippy::unnecessary-lazy-evaluations` implied by `-D warnings`
    mod e {
        pub struct E;
    }
    let _ = Ok(1).unwrap_or_else(|e::E| 2);
    //~^ ERROR: unnecessary closure used to substitute value for `Result::Err`
    let _ = Ok(1).unwrap_or_else(|SomeStruct { .. }| 2);
    //~^ ERROR: unnecessary closure used to substitute value for `Result::Err`

    // Fix #6343
    let arr = [(Some(1),)];
    Some(&0).and_then(|&i| arr[i].0);
}

fn issue11672() {
    // Return type annotation helps type inference and removing it can break code
    let _ = true.then(|| -> &[u8] { &[] });
}
