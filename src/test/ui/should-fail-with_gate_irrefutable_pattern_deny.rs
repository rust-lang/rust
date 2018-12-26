#![feature(irrefutable_let_patterns)]

// should-fail-irrefutable_let_patterns_with_gate
fn main() {
    if let _ = 5 {}
    //~^ ERROR irrefutable if-let pattern [irrefutable_let_patterns]
}
