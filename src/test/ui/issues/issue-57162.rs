// build-pass (FIXME(62277): could be check-pass?)

trait Foo {}
impl Foo for dyn Send {}

impl<T: Sync + Sync> Foo for T {}
fn main() {}
