// This test ensures that the same link is generated in both intra-doc links
// and in jump to def links.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

// First we check intra-doc links.
//@ has 'foo/struct.Bar.html'
//@ has - '//a[@href="{{channel}}/core/fmt/macros/derive.Debug.html"]' 'Debug'
//@ has - '//a[@href="{{channel}}/core/cmp/derive.PartialEq.html"]' 'PartialEq'


// Then we check that they are the same in jump to def.

/// [Debug][derive@Debug] and [PartialEq][derive@PartialEq]
//@ has 'src/foo/jump-to-def-intra-doc.rs.html'
//@ has - '//a[@href="{{channel}}/core/fmt/macros/derive.Debug.html"]' 'Debug'
//@ has - '//a[@href="{{channel}}/core/cmp/derive.PartialEq.html"]' 'PartialEq'
#[derive(Debug, PartialEq)]
pub struct Bar;
