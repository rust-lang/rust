mod rustfmt {}

#[rustfmt::skip] //~ ERROR failed to resolve: could not find `skip` in `rustfmt`
fn main() {}
