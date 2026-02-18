// Test that using rustc_clean/dirty/if_this_changed/then_this_would_need
// are forbidden when `-Zretain-dep-graph` is not enabled.

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

#[rustc_clean(cfg = "foo")] //~ ERROR attribute requires `-Zretain-dep-graph`
fn main() {}

#[rustc_if_this_changed] //~ ERROR attribute requires `-Zretain-dep-graph`
struct Foo<T> {
    f: T,
}

#[rustc_clean(cfg = "foo")] //~ ERROR attribute requires `-Zretain-dep-graph`
type TypeAlias<T> = Foo<T>;

#[rustc_then_this_would_need(variances_of)] //~ ERROR attribute requires `-Zretain-dep-graph`
trait Use<T> {}
