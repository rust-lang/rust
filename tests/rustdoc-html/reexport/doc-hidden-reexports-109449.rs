// Test to enforce rules over re-exports inlining from
// <https://github.com/rust-lang/rust/issues/109449>.

#![crate_name = "foo"]

mod private_module {
    #[doc(hidden)]
    pub struct Public;
    #[doc(hidden)]
    pub type Bar = ();
}

#[doc(hidden)]
mod module {
    pub struct Public2;
    pub type Bar2 = ();
}

#[doc(hidden)]
pub type Bar3 = ();
#[doc(hidden)]
pub struct FooFoo;

// Checking that re-exporting a `#[doc(hidden)]` item will NOT inline it.
pub mod single_reexport {
    //@ has 'foo/single_reexport/index.html'

    // First we check that we have 4 type aliases.
    //@ count - '//*[@id="main-content"]/*[@class="item-table reexports"]//code' 0

    // Then we check that we have the correct link for each re-export.

    //@ !has - '//*[@href="struct.Foo.html"]' 'Foo'
    //@ !has - '//*[@id="reexport.Foo"]/code' 'pub use crate::private_module::Public as Foo;'
    pub use crate::private_module::Public as Foo;
    //@ !has - '//*[@href="type.Foo2.html"]' 'Foo2'
    //@ !has - '//*[@id="reexport.Foo2"]/code' 'pub use crate::private_module::Bar as Foo2;'
    pub use crate::private_module::Bar as Foo2;
    //@ !has - '//*[@href="type.Yo.html"]' 'Yo'
    //@ !has - '//*[@id="reexport.Yo"]/code' 'pub use crate::Bar3 as Yo;'
    pub use crate::Bar3 as Yo;
    //@ !has - '//*[@href="struct.Yo2.html"]' 'Yo2'
    //@ !has - '//*[@id="reexport.Yo2"]/code' 'pub use crate::FooFoo as Yo2;'
    pub use crate::FooFoo as Yo2;

    // Checking that each file is also created as expected.
    //@ !has 'foo/single_reexport/struct.Foo.html'
    //@ !has 'foo/single_reexport/type.Foo2.html'
    //@ !has 'foo/single_reexport/type.Yo.html'
    //@ !has 'foo/single_reexport/struct.Yo2.html'
}

// However, re-exporting an item inheriting `#[doc(hidden)]` will inline it.
pub mod single_reexport_inherit_hidden {
    //@ has 'foo/single_reexport_inherit_hidden/index.html'

    //@ has - '//*[@href="struct.Foo3.html"]' 'Foo3'
    pub use crate::module::Public2 as Foo3;
    //@ has - '//*[@href="type.Foo4.html"]' 'Foo4'
    pub use crate::module::Bar2 as Foo4;

    //@ has 'foo/single_reexport_inherit_hidden/struct.Foo3.html'
    //@ has 'foo/single_reexport_inherit_hidden/type.Foo4.html'
}

pub mod single_reexport_no_inline {
    // First we ensure that we only have re-exports and no inlined items.
    //@ has 'foo/single_reexport_no_inline/index.html'
    //@ count - '//*[@id="main-content"]/*[@class="section-header"]' 1
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Re-exports'

    // Now we check that we don't have links to the items, just `pub use`.
    //@ !has - '//*[@id="main-content"]//*' 'pub use crate::private_module::Public as XFoo;'
    //@ !has - '//*[@id="main-content"]//a' 'XFoo'
    #[doc(no_inline)]
    pub use crate::private_module::Public as XFoo;
    //@ !has - '//*[@id="main-content"]//*' 'pub use crate::private_module::Bar as Foo2;'
    //@ !has - '//*[@id="main-content"]//a' 'Foo2'
    #[doc(no_inline)]
    pub use crate::private_module::Bar as Foo2;
    //@ !has - '//*[@id="main-content"]//*' 'pub use crate::Bar3 as Yo;'
    //@ !has - '//*[@id="main-content"]//a' 'Yo'
    #[doc(no_inline)]
    pub use crate::Bar3 as Yo;
    //@ !has - '//*[@id="main-content"]//*' 'pub use crate::FooFoo as Yo2;'
    //@ !has - '//*[@id="main-content"]//a' 'Yo2'
    #[doc(no_inline)]
    pub use crate::FooFoo as Yo2;
    //@ has - '//*[@id="main-content"]//*' 'pub use crate::module::Public2 as Foo3;'
    //@ !has - '//*[@id="main-content"]//a' 'Foo3'
    #[doc(no_inline)]
    pub use crate::module::Public2 as Foo3;
    //@ has - '//*[@id="main-content"]//*' 'pub use crate::module::Bar2 as Foo4;'
    //@ !has - '//*[@id="main-content"]//a' 'Foo4'
    #[doc(no_inline)]
    pub use crate::module::Bar2 as Foo4;
}

// Checking that glob re-exports don't inline `#[doc(hidden)]` items.
pub mod glob_reexport {
    // With glob re-exports, we don't inline `#[doc(hidden)]` items so only `module` items
    // should be inlined.
    //@ has 'foo/glob_reexport/index.html'
    //@ count - '//*[@id="main-content"]/*[@class="section-header"]' 3
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Re-exports'
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Structs'
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Type Aliases'

    // Now we check we have 1 re-export and 2 inlined items.
    // If not item from a glob re-export is visible, we don't show the re-export.
    //@ !has - '//*[@id="main-content"]//*' 'pub use crate::private_module::*;'
    pub use crate::private_module::*;
    //@ has - '//*[@id="main-content"]//*' 'pub use crate::*;'
    pub use crate::*;
    // This one should be inlined.
    //@ !has - '//*[@id="main-content"]//*' 'pub use crate::module::*;'
    //@ has - '//*[@id="main-content"]//a[@href="struct.Public2.html"]' 'Public2'
    //@ has - '//*[@id="main-content"]//a[@href="type.Bar2.html"]' 'Bar2'
    // And we check that the two files were created too.
    //@ has 'foo/glob_reexport/struct.Public2.html'
    //@ has 'foo/glob_reexport/type.Bar2.html'
    pub use crate::module::*;
}

mod private {
    /// Original.
    pub struct Bar3;
}

// Checking that `#[doc(hidden)]` re-exports documentation isn't generated.
pub mod doc_hidden_reexport {
    //@ has 'foo/doc_hidden_reexport/index.html'
    // Ensure there is only one item in this page and that it's a struct.
    //@ count - '//dt' 1
    //@ has - '//a[@class="struct"]' 'Reexport'
    // Check that the `#[doc(hidden)]` re-export's attributes are not taken into account.
    //@ has - '//dd' 'Visible. Original.'
    /// Visible.
    pub use self::Bar3 as Reexport;
    /// Hidden.
    #[doc(hidden)]
    pub use crate::private::Bar3;
}
