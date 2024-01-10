trait Foo: Send {}

trait Bar {}

impl Bar for dyn Foo {}
impl Bar for dyn Foo + Send {}
//~^ ERROR conflicting implementations of trait `Bar` for type `(dyn Foo + 'static)`

fn main() {}
