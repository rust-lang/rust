// check-pass

#![feature(inline_const)]
#![allow(incomplete_features)]

pub fn todo<T>() -> T {
    const { todo!() }
}

fn main() {
    let _: usize = const { 0 };
}
