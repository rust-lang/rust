trait Foo {}
#[derive(Foo::Anything)] //~ ERROR cannot find
                         //~| ERROR cannot find
struct S;

fn main() {}
