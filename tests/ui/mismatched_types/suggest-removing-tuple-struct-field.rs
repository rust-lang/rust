//@ run-rustfix

macro_rules! my_wrapper {
    ($expr:expr) => { MyWrapper($expr) }
}

pub struct MyWrapper(#[allow(dead_code)] u32);

fn main() {
    let value = MyWrapper(123);
    some_fn(value.0); //~ ERROR mismatched types
    some_fn(my_wrapper!(123).0); //~ ERROR mismatched types
}

fn some_fn(wrapped: MyWrapper) {
    drop(wrapped);
}
