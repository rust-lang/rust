mod foo {
    pub struct Foo;
}

mod bar {
    pub trait Bar {}

    pub fn bar() -> Box<Bar> {
        unimplemented!()
    }
}

// This makes the publicly accessible path
// differ from the internal one.
pub use bar::{Bar, bar};
pub use foo::Foo;
