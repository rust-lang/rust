// Tests that code generated from an external macro (MBE and proc-macro) that
// has an RPIT will not fail when the call-site is 2024.
// https://github.com/rust-lang/rust/issues/132917

//@ proc-macro: no-use-pm.rs
//@ aux-crate: no_use_macro=no-use-macro.rs
//@ edition: 2024
//@ check-pass
//@ ignore-backends: gcc

no_use_pm::pm_rpit!{}

no_use_macro::macro_rpit!{}

fn main() {
    let mut x = vec![];
    x.push(1);

    let element = test_pm(&x);
    x.push(2);
    println!("{element}");

    let element = test_mbe(&x);
    x.push(2);
    println!("{element}");
}
