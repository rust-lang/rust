// ignore-tidy-linelength

#![crate_name = "foo"]
#![feature(doc_cfg)]

// @has 'foo/struct.Foo.html'
// @has '-' '//*[@class="stab portability"]' 'This is supported on feature="sync" only.'
#[doc(cfg(feature = "sync"))]
#[doc(cfg(feature = "sync"))]
pub struct Foo;

// @has 'foo/bar/struct.Bar.html'
// @has '-' '//*[@class="stab portability"]' 'This is supported on feature="sync" only.'
#[doc(cfg(feature = "sync"))]
pub mod bar {
    #[doc(cfg(feature = "sync"))]
    pub struct Bar;
}

// @has 'foo/baz/struct.Baz.html'
// @has '-' '//*[@class="stab portability"]' 'This is supported on feature="sync" and feature="send" only.'
#[doc(cfg(all(feature = "sync", feature = "send")))]
pub mod baz {
    #[doc(cfg(feature = "sync"))]
    pub struct Baz;
}

// @has 'foo/qux/struct.Qux.html'
// @has '-' '//*[@class="stab portability"]' 'This is supported on feature="sync" and feature="send" only.'
#[doc(cfg(feature = "sync"))]
pub mod qux {
    #[doc(cfg(all(feature = "sync", feature = "send")))]
    pub struct Qux;
}

// @has 'foo/quux/struct.Quux.html'
// @has '-' '//*[@class="stab portability"]' 'This is supported on feature="sync" and feature="send" and foo and bar only.'
#[doc(cfg(all(feature = "sync", feature = "send", foo)))]
pub mod quux {
    #[doc(cfg(all(feature = "send", feature = "sync", bar)))]
    pub struct Quux;
}
