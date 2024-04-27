#[doc(hidden)]
pub mod foo {
    pub struct Foo;
}

mod bar {
    pub use self::bar::Bar;
    mod bar {
        pub struct Bar;
    }
}

// @has issue_28537/struct.Foo.html
pub use foo::Foo;

// @has issue_28537/struct.Bar.html
pub use self::bar::Bar;
