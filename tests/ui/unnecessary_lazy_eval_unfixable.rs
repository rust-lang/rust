#![warn(clippy::unnecessary_lazy_evaluations)]

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
}
