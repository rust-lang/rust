use Trait::foo;  //~ ERROR cannot import from a trait or type implementation
//~^ ERROR failed to resolve import
use Foo::new;    //~ ERROR cannot import from a trait or type implementation
//~^ ERROR failed to resolve import

pub trait Trait {
    fn foo();
}

struct Foo;

impl Foo {
    fn new() {}
}

fn main() {}
