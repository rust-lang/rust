pub use foo::bar as foobar;
pub use foobar::baz::*;

pub mod foo {
    pub mod bar {
        pub mod baz {}
    }
}
