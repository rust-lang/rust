// Ensure keyword docs are present with --document-private-items

//@ compile-flags: --document-private-items
#![feature(rustdoc_internals)]

//@ !has "$.index[?(@.name=='match')]"
//@ has  "$.index[?(@.name=='foo')]"
//@ is   "$.index[?(@.name=='foo')].attrs" '[{"content": "#[doc(keyword = \"match\")]", "is_inner": false}]'
//@ is   "$.index[?(@.name=='foo')].docs" '"this is a test!"'
#[doc(keyword = "match")]
/// this is a test!
pub mod foo {}

//@ !has "$.index[?(@.name=='break')]"
//@ has "$.index[?(@.name=='bar')]"
//@ is   "$.index[?(@.name=='bar')].attrs" '[{"content": "#[doc(keyword = \"break\")]", "is_inner": false}]'
//@ is   "$.index[?(@.name=='bar')].docs" '"hello"'
#[doc(keyword = "break")]
/// hello
mod bar {}
