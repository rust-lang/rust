trait Foo {}
#[derive(Foo::Anything)] //~ ERROR cannot find item `Anything`
                         //~| ERROR cannot find item `Anything`
struct S;

fn main() {}
