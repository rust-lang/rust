#![feature(fn_traits)]

// Regression test for https://github.com/rust-lang/rust/issues/128848

fn f<T>(a: T, b: T, c: T) {
    f.call_once()
    //~^ ERROR this method takes 1 argument but 0 arguments were supplied
}

fn main() {}
