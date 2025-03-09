//@ known-bug: #137874
fn a() {
    match b { deref !(0c) };
}
