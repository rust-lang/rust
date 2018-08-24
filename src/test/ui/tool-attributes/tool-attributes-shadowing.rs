mod rustfmt {}

#[rustfmt::skip] //~ ERROR failed to resolve. Could not find `skip` in `rustfmt`
fn main() {}
