//! regression test for <https://github.com/rust-lang/rust/issues/17994>
trait Tr {}
type Huh<T> where T: Tr = isize; //~ ERROR type parameter `T` is never used
fn main() {}
