// Test that methods from shadowed traits can be used

//@ check-pass

mod foo {
    pub trait T { fn f(&self) {} }
    impl T for () {}
}

mod bar { pub use crate::foo::T; }

fn main() {
    pub use bar::*;
    struct T;
    ().f() // OK
}
