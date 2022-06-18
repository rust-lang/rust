trait Foo {}
#[derive(Foo::Anything)] //~ ERROR failed to resolve: partially resolved path in a derive macro
                         //~| ERROR failed to resolve: partially resolved path in a derive macro
struct S;

fn main() {}
