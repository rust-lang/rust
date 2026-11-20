//@ aux-build:all-item-types.rs

// This test is to ensure there is no problem on handling foreign functions
// coming from a dependency.

#![crate_name = "foo"]

extern crate all_item_types;

//@ has 'foo/fn.foo_ffn.html'
//@ has - '//*[@class="rust item-decl"]//code' 'pub unsafe extern "C" fn foo_ffn()'
pub use all_item_types::foo_ffn;
