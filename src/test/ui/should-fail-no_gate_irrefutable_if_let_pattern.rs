// should-fail-irrefutable_let_patterns
fn main() {
    if let _ = 5 {}
    //~^ ERROR irrefutable if-let pattern [E0162]
}
