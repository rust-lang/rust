// check-pass

pub const fn f<T, const N: usize>(_: [std::mem::MaybeUninit<T>; N]) {}

fn main() {}
