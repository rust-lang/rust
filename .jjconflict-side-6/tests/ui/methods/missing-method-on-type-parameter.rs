//@ edition:2015..2021
// Regression test for https://github.com/rust-lang/rust/issues/129205
fn x<T: Copy>() {
    T::try_from(); //~ ERROR E0599
}

fn main() {}
