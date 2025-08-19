// ensures that 'use foo:*' doesn't import non-public item

use m1::*;

mod foo {
    pub fn foo() {}
}
mod a {
    pub mod b {
        use crate::foo::foo;
        type Bar = isize;
    }
    pub mod sub {
        use crate::a::b::*;
        fn sub() -> Bar { 1 }
        //~^ ERROR cannot find type `Bar` in this scope
    }
}

mod m1 {
    fn foo() {}
}

fn main() {
    foo(); //~ ERROR expected function, found module `foo`
}
