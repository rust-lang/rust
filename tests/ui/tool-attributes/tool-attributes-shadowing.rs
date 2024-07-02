mod rustfmt {}

#[rustfmt::skip] //~ ERROR cannot find macro `skip`
fn main() {}
