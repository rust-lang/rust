// Regression test for issue https://github.com/rust-lang/rust/issues/28586

pub trait Foo {}
impl Foo for [u8; usize::BYTES] {}
//~^ ERROR no associated function or constant named `BYTES` found

fn main() { }
