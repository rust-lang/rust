pub mod one {
    mod foo {
        pub struct Foo;
    }

    pub use self::foo::Foo;
}

pub mod two {
    mod foo {
        mod bar {
            pub struct Foo;
        }
    }

    pub use crate::two::foo::Foo; //~ ERROR unresolved import `crate::two::foo::Foo` [E0432]
}

fn main() {}
