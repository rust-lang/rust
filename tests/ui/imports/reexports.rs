#![warn(unused_imports)]

mod a {
    fn foo() {}
    mod foo {}

    mod a {
        pub use super::foo;
        //~^ ERROR cannot be re-exported
        //~| WARNING unused import: `super::foo`
        pub use super::*;
        //~^ WARNING glob import doesn't reexport anything with visibility `pub` because no imported item is public enough
        //~| WARNING unused import: `super::*`
    }
}

mod b {
    pub fn foo() {}
    mod foo {
        pub struct S;
    }

    pub mod a {
        pub use super::foo; // This is OK since the value `foo` is visible enough.
        fn f(_: foo::S) {} // `foo` is imported in the type namespace (but not `pub` re-exported).
    }

    pub mod b {
        pub use super::*; // This is also OK since the value `foo` is visible enough.
        fn f(_: foo::S) {} // Again, the module `foo` is imported (but not `pub` re-exported).
    }
}

mod c {
    // Test that `foo` is not re-exported.
    use crate::b::a::foo::S; //~ ERROR `foo`
    use crate::b::b::foo::S as T; //~ ERROR `foo`
}

fn main() {}
