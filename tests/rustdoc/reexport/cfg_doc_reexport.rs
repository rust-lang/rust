#![feature(doc_cfg)]
#![feature(no_core, lang_items)]

#![crate_name = "foo"]
#![no_core]

#[lang = "sized"]
trait Sized {}

//@ has 'foo/index.html'
//@ has - '//dt/*[@class="stab portability"]' 'foobar'
//@ has - '//dt/*[@class="stab portability"]' 'bar'

#[doc(cfg(feature = "foobar"))]
mod imp_priv {
    //@ has 'foo/struct.BarPriv.html'
    //@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
    //    'Available on crate feature foobar only.'
    pub struct BarPriv {}
    impl BarPriv {
        pub fn test() {}
    }
}
#[doc(cfg(feature = "foobar"))]
pub use crate::imp_priv::*;

pub mod bar {
    //@ has 'foo/bar/struct.Bar.html'
    //@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
    //    'Available on crate feature bar only.'
    #[doc(cfg(feature = "bar"))]
    pub struct Bar;
}

#[doc(cfg(feature = "bar"))]
pub use bar::Bar;
