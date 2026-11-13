//@ known-bug: #144241
fn main() {
    |_: dyn ?Sized + !Send| {}
}
