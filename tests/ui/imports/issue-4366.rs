// regression test for issue 4366

// ensures that 'use foo:*' doesn't import non-public 'use' statements in the
// module 'foo'

use m1::*;

mod foo {
    pub fn foo() {}
}
mod a {
    pub mod b {
        use foo::foo;
        type Bar = isize;
    }
    pub mod sub {
        use a::b::*;
        fn sub() -> isize { foo(); 1 } //~ ERROR cannot find function `foo` in this scope
    }
}

mod m1 {
    fn foo() {}
}

fn main() {}
