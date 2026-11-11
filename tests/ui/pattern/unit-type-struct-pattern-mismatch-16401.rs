//! Regression test for https://github.com/rust-lang/rust/issues/16401

struct Slice<T> {
    data: *const T,
    len: usize,
}

fn main() {
    match () { //~ NOTE this expression has type `()`
        Slice { data: data, len: len } => (),
        //~^ ERROR mismatched types
        //~| NOTE expected unit type `()`
        //~| NOTE found struct `Slice<_>`
        //~| NOTE expected `()`, found `Slice<_>`
        _ => unreachable!()
    }
}
