//@ known-bug: #134217

impl<A> std::ops::CoerceUnsized<A> for A {}

fn main() {
    if let _ = true
        && true
    {}
}
