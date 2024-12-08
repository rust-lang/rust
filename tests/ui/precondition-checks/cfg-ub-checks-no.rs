//@ run-pass
//@ compile-flags: -Zub-checks=no

#![feature(cfg_ub_checks)]

fn main() {
    assert!(!cfg!(ub_checks));
    assert!(compiles_differently());
}

#[cfg(ub_checks)]
fn compiles_differently() -> bool {
    false
}

#[cfg(not(ub_checks))]
fn compiles_differently() -> bool {
    true
}
