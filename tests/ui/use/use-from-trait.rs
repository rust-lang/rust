use Trait::foo; //~ ERROR `foo` is not directly importable
use Trait::Assoc; //~ ERROR `Assoc` is not directly importable
use Trait::C; //~ ERROR `C` is not directly importable

use Foo::new; //~ ERROR unresolved import `Foo` [E0432]

use Foo::C2; //~ ERROR unresolved import `Foo` [E0432]

pub trait Trait {
    fn foo();
    type Assoc;
    const C: u32;
}

struct Foo;

impl Foo {
    fn new() {}
    const C2: u32 = 0;
}

fn main() {}
