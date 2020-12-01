// From <https://github.com/rust-lang/rust/issues/77361>

// edition:2018

const _: i32 = { core::mem::ManuallyDrop::new(async { 0 }); 4 };
//~^ `async` block

fn main() {}
