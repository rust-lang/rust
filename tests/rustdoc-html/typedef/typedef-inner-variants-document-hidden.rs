// Regression test for <https://github.com/rust-lang/rust/issues/144969>.

//@ compile-flags: -Z unstable-options --document-hidden-items
//@ aux-build:cross_crate_generic_typedef.rs

#![crate_name = "document_hidden_aliased_items"]

extern crate cross_crate_generic_typedef;

use std::collections::BTreeMap;

//@ has 'document_hidden_aliased_items/type.InlineU64.html'
//@ count - '//*[@class="structfield section-header"]' 2
//@ has - '//pre[@class="rust item-decl"]//code' 'pub hidden'
//@ count - '//pre[@class="rust item-decl"]//code' '/* private fields */' 0
pub type InlineU64 = cross_crate_generic_typedef::InlineOne<u64>;

//@ has 'document_hidden_aliased_items/type.InlineEnum.html'
//@ count - '//*[@class="variant"]' 3
//@ has - '//pre[@class="rust item-decl"]//code' 'Hidden'
//@ count - '//pre[@class="rust item-decl"]//code' '// some variants omitted' 0
pub type InlineEnum = cross_crate_generic_typedef::GenericEnum<i32>;

//@ has 'document_hidden_aliased_items/type.PrivateFields.html'
//@ has - '//*[@class="rust item-decl"]/code' 'struct PrivateFields { /* private fields */ }'
pub type PrivateFields = BTreeMap<u32, String>;
