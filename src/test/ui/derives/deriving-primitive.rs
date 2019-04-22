#[derive(FromPrimitive)] //~ ERROR cannot find derive macro `FromPrimitive` in this scope
enum Foo {}

fn main() {}
