// Regression test for https://github.com/rust-lang/rust/issues/100973

#![feature(no_core)]
#![no_core]

// @set m1 = "$.index[*][?(@.name == 'm1' && @.kind == 'module')].id"
// @is "$.index[*][?(@.name == 'm1' && @.kind == 'module')].inner.items" []
// @is "$.index[*][?(@.name == 'm1' && @.kind == 'module')].inner.is_stripped" true
mod m1 {
    pub fn f() {}
}
// @set m2 = "$.index[*][?(@.name == 'm2' && @.kind == 'module')].id"
// @is "$.index[*][?(@.name == 'm2' && @.kind == 'module')].inner.items" []
// @is "$.index[*][?(@.name == 'm2' && @.kind == 'module')].inner.is_stripped" true
mod m2 {
    pub fn f(_: u8) {}
}

// @set m1_use = "$.index[*][?(@.inner.name=='m1')].id"
// @is "$.index[*][?(@.inner.name=='m1')].inner.id" $m1
// @is "$.index[*][?(@.inner.name=='m1')].inner.glob" true
pub use m1::*;
// @set m2_use = "$.index[*][?(@.inner.name=='m2')].id"
// @is "$.index[*][?(@.inner.name=='m2')].inner.id" $m2
// @is "$.index[*][?(@.inner.name=='m2')].inner.glob" true
pub use m2::*;

// @ismany "$.index[*][?(@.inner.is_crate==true)].inner.items[*]" $m1_use $m2_use
