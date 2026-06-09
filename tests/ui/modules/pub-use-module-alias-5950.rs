// https://github.com/rust-lang/rust/issues/5950
//@ check-pass

pub use local as local_alias;

pub mod local { }

pub fn main() {}
