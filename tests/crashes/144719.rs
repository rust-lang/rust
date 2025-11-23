//@ known-bug: rust-lang/rust#144719
#[derive(PartialEq)]
struct Thing(&'static Thing);

static X: Thing = Thing(&X);

const Y: &Thing = &X;

fn main() {
    if let Y = Y {}
}
