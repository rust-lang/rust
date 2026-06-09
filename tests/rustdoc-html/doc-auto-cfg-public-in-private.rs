// This test ensures that even though private items are removed from generated docs,
// their `cfg`s will still impact their child items.

#![feature(doc_cfg)]
#![crate_name = "foo"]

pub struct X;

#[cfg(not(feature = "blob"))]
fn foo() {
    impl X {
        //@ has 'foo/struct.X.html'
        //@ has - '//*[@class="stab portability"]' 'Available on non-crate feature blob only.'
        pub fn bar() {}
    }
}
