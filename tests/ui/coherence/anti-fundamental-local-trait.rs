//@ check-pass

// Test that `#[rustc_anti_fundamental]` does NOT block local traits.
// If the trait itself is local, orphan rules pass even on fundamental types.

#![feature(fundamental)]
#![feature(rustc_attrs)]

#[fundamental]
struct LocalFundamental<T>(T);

#[rustc_anti_fundamental]
trait AntiFundamentalTrait {}

struct LocalType;

// OK: both trait and fundamental type are local.
impl AntiFundamentalTrait for LocalFundamental<LocalType> {}

// OK: implementing on a local non-fundamental type.
impl AntiFundamentalTrait for LocalType {}

fn main() {}
