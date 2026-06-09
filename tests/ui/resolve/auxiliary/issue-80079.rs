#![crate_type = "lib"]

pub mod public {
    use crate::private_import;

    // should not be suggested since it is private
    struct Foo;

    mod private_module {
        // should not be suggested since it is private
        pub struct Foo;
    }
}

mod private_import {
    // should not be suggested since it is private
    pub struct Foo;
}
