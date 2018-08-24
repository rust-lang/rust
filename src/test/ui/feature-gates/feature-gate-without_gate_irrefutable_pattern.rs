// gate-test-irrefutable_let_patterns


#[allow(irrefutable_let_patterns)]
fn main() {
    if let _ = 5 {}
    //~^ ERROR 15:12: 15:13: irrefutable if-let pattern [E0162]
}
