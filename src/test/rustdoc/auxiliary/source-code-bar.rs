//! just some other file. :)

use crate::Foo;

pub struct Bar {
    field: Foo,
}

pub struct Bar2 {
    field: crate::Foo,
}

pub mod sub {
    pub trait Trait {
        fn tadam() {}
    }
}
