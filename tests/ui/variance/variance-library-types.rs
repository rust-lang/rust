// Check that certain library types have correct variance.
#![feature(rustc_attrs)]


#[rustc_dump_variances]
struct LazyCellIsCovariantOnlyInF<T, F>(std::cell::LazyCell<T, F>); //~ ERROR [T: o, F: +]

#[rustc_dump_variances]
struct LazyLockIsCovariantOnlyInF<T, F>(std::sync::LazyLock<T, F>); //~ ERROR [T: o, F: +]

pub fn main() {}
