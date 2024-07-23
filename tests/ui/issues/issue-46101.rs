trait Foo {}
#[derive(Foo::Anything)] //~ ERROR cannot find macro `Anything`
                         //~| ERROR cannot find macro `Anything`
struct S;

fn main() {}
