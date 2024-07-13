// https://github.com/rust-lang/rust/issues/106142
#![crate_name="foo"]

//@ has 'foo/a/index.html'
//@ count 'foo/a/index.html' '//ul[@class="item-table"]//li//a' 1

#![allow(rustdoc::broken_intra_doc_links)]

pub mod a {
    /// [`m`]
    pub fn f() {}

    #[macro_export]
    macro_rules! m {
        () => {};
    }
}
