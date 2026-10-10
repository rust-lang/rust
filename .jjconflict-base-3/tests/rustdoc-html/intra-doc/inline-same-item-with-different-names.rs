// This test ensures that when a same item is inlined with different names, the intra
// doc links generate the correct href/title.
// Regression test for <https://github.com/rust-lang/rust/issues/136777>.

#![crate_name = "foo"]

// We check that the macros and structs are correctly generated.
//@ has 'foo/macro.d1.html'
//@ has 'foo/macro.d2.html'
//@ has 'foo/macro.d3.html'
//@ has 'foo/struct.a1.html'
//@ has 'foo/struct.a2.html'
//@ has 'foo/struct.a3.html'

//@ has 'foo/index.html'

//@ has - '//dd/a[@href="macro.d1.html"]' 'd1'
//@ has - '//dd/a[@title="macro foo::d1"]' 'd1'
//@ has - '//dd/a[@href="macro.d2.html"]' 'd2'
//@ has - '//dd/a[@title="macro foo::d2"]' 'd2'
//@ has - '//dd/a[@href="macro.d3.html"]' 'd3'
//@ has - '//dd/a[@title="macro foo::d3"]' 'd3'

/// Link to [`d3`].
pub use std::debug_assert as d1;
/// Link to [`d1`].
pub use std::debug_assert as d2;
/// Link to [`d2`].
pub use std::debug_assert as d3;

//@ has - '//dd/a[@href="struct.a1.html"]' 'a1'
//@ has - '//dd/a[@title="struct foo::a1"]' 'a1'
//@ has - '//dd/a[@href="struct.a2.html"]' 'a2'
//@ has - '//dd/a[@title="struct foo::a2"]' 'a2'
//@ has - '//dd/a[@href="struct.a3.html"]' 'a3'
//@ has - '//dd/a[@title="struct foo::a3"]' 'a3'

/// Link to [`a3`].
pub use std::ffi::os_str::OsString as a1;
/// Link to [`a1`].
pub use std::ffi::os_str::OsString as a2;
/// Link to [`a2`].
pub use std::ffi::os_str::OsString as a3;
