// Test for <https://github.com/rust-lang/rust/issues/157755>.

//@ check-pass

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

#[derive(Clone)]
pub struct Local;

pub type Alias = Local;
impl Copy for Alias {}

mod drop_impl {
    pub struct Local;

    pub type Alias = Local;
    impl Drop for Alias {
        fn drop(&mut self) {}
    }
}

fn main() {}
