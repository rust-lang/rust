// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Z unpretty=expanded

fn main() {
    if let 0 = 1 {}
}
