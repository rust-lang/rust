// Regression test for https://github.com/rust-lang/rust/issues/100973

//@ set m1 = "$.index[?(@.name == 'm1' && @.inner.module)].id"
//@ is "$.index[?(@.name == 'm1')].inner.module.items" []
//@ is "$.index[?(@.name == 'm1')].inner.module.is_stripped" true
mod m1 {
    pub fn f() {}
}
//@ set m2 = "$.index[?(@.name == 'm2' && @.inner.module)].id"
//@ is "$.index[?(@.name == 'm2')].inner.module.items" []
//@ is "$.index[?(@.name == 'm2')].inner.module.is_stripped" true
mod m2 {
    pub fn f(_: u8) {}
}

//@ set m1_use = "$.index[?(@.docs=='m1 re-export')].id"
//@ is "$.index[?(@.inner.use.name=='m1')].inner.use.id" $m1
//@ is "$.index[?(@.inner.use.name=='m1')].inner.use.is_glob" true
/// m1 re-export
pub use m1::*;
//@ set m2_use = "$.index[?(@.docs=='m2 re-export')].id"
//@ is "$.index[?(@.inner.use.name=='m2')].inner.use.id" $m2
//@ is "$.index[?(@.inner.use.name=='m2')].inner.use.is_glob" true
/// m2 re-export
pub use m2::*;

//@ ismany "$.index[?(@.name=='glob_collision')].inner.module.items[*]" $m1_use $m2_use
