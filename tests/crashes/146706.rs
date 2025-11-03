// This is part of series of regression tests for some diagnostics ICEs encountered in the wild with
// suggestions having overlapping parts under https://github.com/rust-lang/rust/pull/146121.

//@ needs-rustc-debug-assertions
//@ known-bug: #146706

type Alias<'a, T> = Foo<T>;

enum Foo<T> {
    Bar { t: T },
}

fn main() {
    Alias::Bar::<u32> { t: 0 };
}
