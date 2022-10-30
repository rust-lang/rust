// aux-build:pub-struct.rs
// ignore-tidy-linelength

// Test for the ICE in https://github.com/rust-lang/rust/issues/83057
// An external type re-exported with different attributes shouldn't cause an error

#![no_core]
#![feature(no_core)]

extern crate pub_struct as foo;
#[doc(inline)]

// @set crate_use_id = "$.index[*][?(@.docs=='Hack A')].id"
// @set foo_id = "$.index[*][?(@.docs=='Hack A')].inner.id"
/// Hack A
pub use foo::Foo;

// @set bar_id = "$.index[*][?(@.name=='bar')].id"
pub mod bar {
    // @is "$.index[*][?(@.docs=='Hack B')].inner.id" $foo_id
    // @set bar_use_id = "$.index[*][?(@.docs=='Hack B')].id"
    // @ismany "$.index[*][?(@.name=='bar')].inner.items[*]" $bar_use_id
    /// Hack B
    pub use foo::Foo;
}

// @ismany "$.index[*][?(@.kind=='import')].id" $crate_use_id $bar_use_id
// @ismany "$.index[*][?(@.name=='private_twice_one_inline')].inner.items[*]" $bar_id $crate_use_id
