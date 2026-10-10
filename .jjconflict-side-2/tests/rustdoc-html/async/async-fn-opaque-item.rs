//@ compile-flags: --document-private-items --crate-type=lib
//@ edition: 2021

// Issue 109931 -- test against accidentally documenting the `impl Future`
// that comes from an async fn desugaring.

// Check that we don't document an unnamed opaque type
//@ !has async_fn_opaque_item/opaque..html

// Checking there is only a "Functions" header and no "Opaque types".
//@ has async_fn_opaque_item/index.html
//@ count - '//*[@class="section-header"]' 1
//@ has - '//*[@class="section-header"]' 'Functions'

pub async fn test() {}
