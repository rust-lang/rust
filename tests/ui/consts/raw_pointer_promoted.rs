//@ check-pass

pub const FOO: &'static *const i32 = &(&0 as _);

fn main() {}
