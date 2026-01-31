//@ check-pass
//@ compile-flags: -Cinstrument-coverage -Zno-profiler-runtime
//@ edition: 2024

// Regression test for issue #147339
// Nested macro expansions should not cause ICE during coverage instrumentation

macro_rules! foo {
    ($($m:ident $($f:ident $v:tt)+),*) => {
        $($(macro_rules! $f { () => { $v } })+)*
        $(macro_rules! $m { () => { $(fn $f() -> i32 { $v })+ } })*
    }
}

foo!(m a 1 b 2, n c 3);
m!();
n!();

fn main() {}
