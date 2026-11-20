//! Regression test for https://github.com/rust-lang/rust/issues/46101
trait Foo {}
#[derive(Foo::Anything)] //~ ERROR cannot find
                         //~| ERROR cannot find
struct S;

fn main() {}
