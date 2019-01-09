#[doc(alias = "foo")] //~ ERROR: #[doc(alias = "...")] is experimental
pub struct Foo;

fn main() {}
