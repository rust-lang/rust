// Tests that code generated from an external macro (MBE and proc-macro) that
// has an RPIT will not fail when the call-site is 2024.
// https://github.com/rust-lang/rust/issues/132917

//@ aux-crate: no_use_pm=no-use-pm.rs
//@ aux-crate: no_use_macro=no-use-macro.rs
//@ edition: 2024
//@ compile-flags:-Z unstable-options

no_use_pm::pm_rpit!{}
//~^ ERROR: cannot borrow `x` as mutable

no_use_macro::macro_rpit!{}
//~^ ERROR: cannot borrow `x` as mutable

fn main() {
    let mut x = vec![];
    x.push(1);

    let element = test_pm(&x);
    x.push(2);
    //~^ ERROR: cannot borrow `x` as mutable
    println!("{element}");

    let element = test_mbe(&x);
    x.push(2);
    //~^ ERROR: cannot borrow `x` as mutable
    println!("{element}");
}
