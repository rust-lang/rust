// This test ensures that if a private item is re-exported with an intermediate
// `#[doc(hidden)]` re-export, it'll still be inlined (and not include any attribute
// from the doc hidden re-export.

#![crate_name = "foo"]

//@ has 'foo/index.html'
// There should only be one struct displayed.
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Structs'
//@ has - '//*[@id="main-content"]//a[@href="struct.Reexport.html"]' 'Reexport'
//@ has - '//*[@id="main-content"]//dd' 'Visible. Original.'

mod private {
    /// Original.
    pub struct Bar3;
}

/// Hidden.
#[doc(hidden)]
pub use crate::private::Bar3;
/// Visible.
pub use self::Bar3 as Reexport;
