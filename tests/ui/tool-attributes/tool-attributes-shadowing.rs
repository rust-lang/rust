mod rustfmt {}

#[rustfmt::skip] //~ ERROR: cannot find `skip` in `rustfmt`
fn main() {}
