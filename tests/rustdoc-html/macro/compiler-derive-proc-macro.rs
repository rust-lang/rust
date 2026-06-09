// This test ensures that compiler builtin proc-macros are considered as such.

#![crate_name = "foo"]

//@ has 'foo/index.html'
// Each compiler builtin proc-macro has a trait equivalent so we should have
// a trait section as well.
//@ count - '//*[@id="main-content"]//*[@class="section-header"]' 2
//@ has - '//*[@id="main-content"]//*[@class="section-header"]' 'Traits'
//@ has - '//*[@id="main-content"]//*[@class="section-header"]' 'Derive Macros'

// Now checking the correct file is generated as well.
//@ has 'foo/derive.Clone.html'
//@ !has 'foo/macro.Clone.html'
pub use std::clone::Clone;
