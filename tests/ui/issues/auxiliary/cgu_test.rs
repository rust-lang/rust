//@ no-prefer-dynamic
//@ compile-flags: --crate-type=lib

pub fn id<T>(t: T) -> T {
  t
}
