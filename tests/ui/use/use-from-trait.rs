use Trait::foo; //~ ERROR `use` associated items of traits is unstable [E0658]
use Trait::Assoc; //~ ERROR `use` associated items of traits is unstable [E0658]
use Trait::C; //~ ERROR `use` associated items of traits is unstable [E0658]

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
