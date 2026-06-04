//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/32518
//@ no-prefer-dynamic
//@ compile-flags: --crate-type=lib

pub fn id<T>(t: T) -> T {
  t
}
