#[lang = "foo"] //~ ERROR language items are subject to change
                //~^ ERROR definition of an unknown language item: `foo`
trait Foo {}

fn main() {}
