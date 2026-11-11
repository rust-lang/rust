// https://github.com/rust-lang/rust/issues/57162
//@ check-pass

trait Foo {}
impl Foo for dyn Send {}

impl<T: Sync + Sync> Foo for T {}
fn main() {}
