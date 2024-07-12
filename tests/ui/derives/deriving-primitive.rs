#[derive(FromPrimitive)] //~ ERROR cannot find derive macro `FromPrimitive`
                         //~| ERROR cannot find derive macro `FromPrimitive`
enum Foo {}

fn main() {}
