//! This test makes sure that with never show the inner fields in the
//! aliased type view of type alias.

#![crate_name = "foo"]

use std::collections::BTreeMap;

//@ has 'foo/type.FooBar.html' '//*[@class="rust item-decl"]/code' 'struct FooBar { /* private fields */ }'
pub type FooBar = BTreeMap<u32, String>;
