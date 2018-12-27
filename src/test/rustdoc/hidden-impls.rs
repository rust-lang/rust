#![crate_name = "foo"]

mod hidden {
    #[derive(Clone)]
    pub struct Foo;
}

#[doc(hidden)]
pub mod __hidden {
    pub use hidden::Foo;
}

// @has foo/trait.Clone.html
// @!has - 'Foo'
// @has implementors/foo/trait.Clone.js
// @!has - 'Foo'
pub use std::clone::Clone;
