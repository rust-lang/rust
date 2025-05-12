// Regression test for https://github.com/rust-lang/rust/issues/101103

mod m1 {
    pub fn x() {}
}

pub use m1::x;

//@ has "$.index[?(@.name=='x' && @.inner.function)]"
//@ has "$.index[?(@.inner.use.name=='x')].inner.use.source" '"m1::x"'
//@ !has "$.index[?(@.name=='m1')]"
