// check-pass

#![feature(inline_const)]

pub fn todo<T>() -> T {
    const { todo!() }
}

fn main() {
    let _: usize = const { 0 };
}
