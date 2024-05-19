// Test that methods from shadowed traits cannot be used

mod foo {
    pub trait T { fn f(&self) {} }
    impl T for () {}
}

mod bar { pub use foo::T; }

fn main() {
    pub use bar::*;
    struct T;
    ().f() //~ ERROR no method
}
