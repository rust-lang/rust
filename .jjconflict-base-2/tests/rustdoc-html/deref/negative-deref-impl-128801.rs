#![feature(negative_impls)]
#![crate_name = "foo"]

// Regression test for https://github.com/rust-lang/rust/issues/128801
// Negative `Deref`/`DerefMut` impls should not cause an ICE and should still be rendered.

pub struct Source;

//@ has foo/struct.Source.html

// Verify negative Deref impl is rendered in the main content.
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'impl !Deref for Source'

// Verify negative DerefMut impl is rendered in the main content.
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'impl !DerefMut for Source'

// Verify negative impls appear in the sidebar.
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#trait-implementations"]' 'Trait Implementations'
//@ has - '//*[@class="sidebar-elems"]//section//a' '!Deref'
//@ has - '//*[@class="sidebar-elems"]//section//a' '!DerefMut'

impl !std::ops::Deref for Source {}
impl !std::ops::DerefMut for Source {}
