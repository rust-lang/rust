#[lang = "foo"] //~ ERROR lang items are subject to change
                //~^ ERROR definition of an unknown lang item: `foo`
trait Foo {}

fn main() {}
