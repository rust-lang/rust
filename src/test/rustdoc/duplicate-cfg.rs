#![crate_name = "foo"]
#![feature(doc_cfg)]

// @has 'foo/index.html'
// @!has '-' '//*[@class="stab portability"]' 'feature="sync" and'
// @has '-' '//*[@class="stab portability"]' 'feature="sync"'
#[doc(cfg(feature = "sync"))]
#[doc(cfg(feature = "sync"))]
pub struct Foo;

#[doc(cfg(feature = "sync"))]
pub mod bar {
    #[doc(cfg(feature = "sync"))]
    pub struct Bar;
}
