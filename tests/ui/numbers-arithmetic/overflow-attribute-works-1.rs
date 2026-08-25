//@ run-pass
//@ compile-flags: -C overflow_checks=true

#![feature(cfg_overflow_checks)]

fn main() {
    assert!(cfg!(overflow_checks));
    assert!(compiles_differently());
}

#[cfg(overflow_checks)]
fn compiles_differently()->bool {
    true
}

#[cfg(not(overflow_checks))]
fn compiles_differently()->bool {
    false
}
