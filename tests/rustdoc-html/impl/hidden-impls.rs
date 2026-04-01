#![crate_name = "foo"]

mod hidden {
    #[derive(Clone)]
    pub struct Foo;
}

#[doc(hidden)]
pub mod __hidden {
    pub use hidden::Foo;
}

//@ has foo/trait.Clone.html
//@ !hasraw - 'Foo'
//@ has trait.impl/core/clone/trait.Clone.js
//@ !hasraw - 'Foo'
pub use std::clone::Clone;
