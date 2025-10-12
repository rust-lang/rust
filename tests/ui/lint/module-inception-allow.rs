#![allow(module_inception)]

pub mod foo {
    pub mod foo {
        pub fn bar() {}
    }
}

pub mod baz {
    // This is fine - different name
    pub fn qux() {}
}

pub mod outer {
    pub mod outer {
        pub fn inner() {}
    }

    pub mod different {
        pub fn func() {}
    }
}

fn main() {}
