// ensures that 'use foo:*' doesn't import non-public item

use m1::*;

mod foo {
    pub fn foo() {}
}
mod a {
    pub mod b {
        use foo::foo;
        type bar = isize;
    }
    pub mod sub {
        use a::b::*;
        fn sub() -> bar { 1 }
        //~^ ERROR cannot find type `bar` in this scope
    }
}

mod m1 {
    fn foo() {}
}

fn main() {
    foo(); //~ ERROR expected function, found module `foo`
}
