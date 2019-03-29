// Test that changing what a `type` points to does not go unnoticed
// by the variance analysis.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn main() { }

struct Foo<T> {
    f: T
}

#[rustc_if_this_changed(Krate)]
type TypeAlias<T> = Foo<T>;

#[rustc_then_this_would_need(variances_of)] //~ ERROR OK
struct Use<T> {
    x: TypeAlias<T>
}
