//! regression test for https://github.com/rust-lang/rust/issues/20772
trait T: Iterator<Item = Self::Item> //~ ERROR cycle detected
{
}

fn main() {}
