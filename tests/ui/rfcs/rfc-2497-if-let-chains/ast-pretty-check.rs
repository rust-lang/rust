//@ check-pass
//@ compile-flags: -Z unpretty=expanded
//@ edition: 2015

fn main() {
    if let 0 = 1 {}
}
