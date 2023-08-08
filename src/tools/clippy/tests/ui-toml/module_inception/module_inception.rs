#![warn(clippy::module_inception)]

// Lint
pub mod foo2 {
    pub mod bar2 {
        pub mod bar2 {
            pub mod foo2 {}
        }
        pub mod foo2 {}
    }
    pub mod foo2 {
        pub mod bar2 {}
    }
}

// Don't lint
mod foo {
    pub mod bar {
        pub mod foo {
            pub mod bar {}
        }
    }
    pub mod foo {
        pub mod bar {}
    }
}

// No warning. See <https://github.com/rust-lang/rust-clippy/issues/1220>.
pub mod bar {
    #[allow(clippy::module_inception)]
    pub mod bar {}
}

fn main() {}
