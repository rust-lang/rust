//@ edition: 2021
//@ min-llvm-version: 18
//@ revisions: good bad
//@ check-pass
//@ compile-flags: -Cinstrument-coverage -Zcoverage-options=mcdc -Zno-profiler-runtime

#[cfg(good)]
fn main() {
    let [a, b, c, d, e, f] = <[bool; 6]>::default();
    if a && b && c && d && e && f {
        core::hint::black_box("hello");
    }
}

#[cfg(bad)]
fn main() {
    let [a, b, c, d, e, f, g] = <[bool; 7]>::default();
    if a && b && c && d && e && f && g { //[bad]~ WARNING Conditions number of the decision
        core::hint::black_box("hello");
    }
}
