// Regression test for issue #28586

pub trait Foo {}
impl Foo for [u8; usize::BYTES] {}
//~^ ERROR no associated function or constant named `BYTES` found

fn main() { }
