// Checks that `unused_qualifications` don't fire on explicit global paths.
// Issue: <https://github.com/rust-lang/rust/issues/122374>.

//@ check-pass

#![deny(unused_qualifications)]

pub fn bar() -> u64 {
    ::std::default::Default::default()
}

fn main() {}
