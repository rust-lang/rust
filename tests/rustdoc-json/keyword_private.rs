// Ensure keyword docs are present with --document-private-items

//@ compile-flags: --document-private-items
#![feature(rustdoc_internals)]

// @!has "$.index[*][?(@.name=='match')]"
// @has  "$.index[*][?(@.name=='foo')]"
// @is   "$.index[*][?(@.name=='foo')].attrs" '["#[doc(keyword = \"match\")]"]'
// @is   "$.index[*][?(@.name=='foo')].docs" '"this is a test!"'
#[doc(keyword = "match")]
/// this is a test!
pub mod foo {}

// @!has "$.index[*][?(@.name=='hello')]"
// @has "$.index[*][?(@.name=='bar')]"
// @is   "$.index[*][?(@.name=='bar')].attrs" '["#[doc(keyword = \"hello\")]"]'
// @is   "$.index[*][?(@.name=='bar')].docs" '"hello"'
#[doc(keyword = "hello")]
/// hello
mod bar {}
