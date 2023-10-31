#![warn(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::unnecessary_literal_unwrap)]

struct Deep(Option<usize>);

#[derive(Copy, Clone)]
struct SomeStruct {
    some_field: usize,
}

fn main() {
    // fix will break type inference
    let _ = Ok(1).unwrap_or_else(|()| 2);
    mod e {
        pub struct E;
    }
    let _ = Ok(1).unwrap_or_else(|e::E| 2);
    let _ = Ok(1).unwrap_or_else(|SomeStruct { .. }| 2);

    // Fix #6343
    let arr = [(Some(1),)];
    Some(&0).and_then(|&i| arr[i].0);
}
