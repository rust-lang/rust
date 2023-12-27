// check-pass

macro_rules! a { ($ty:ty) => {} }

a! { for<T = &i32> fn() }

fn main() {}
