// run-pass
#![feature(irrefutable_let_patterns)]

// must-compile-successfully-irrefutable_let_patterns_with_gate
#[allow(irrefutable_let_patterns)]
fn main() {
    if let _ = 5 {}

    while let _ = 5 {
        break;
    }
}
