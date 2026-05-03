// This test ensures that reexports cfgs are correctly computed.
// Regression test for <https://github.com/rust-lang/rust/issues/150268>.

// ignore-tidy-linelength

#![feature(doc_cfg)]
#![crate_name = "foo"]

//@has 'foo/struct.FlatBanana.html'
//@has - '//*[@class="item-info"]/*[@class="stab portability"]' 'Available on non-crate feature banana and non-crate feature yoyo only.'

//@has 'foo/struct.SubBanana.html'
//@has - '//*[@class="item-info"]/*[@class="stab portability"]' 'Available on non-crate feature ananas and non-crate feature banana and non-crate feature yoyo only.'

#[cfg(not(feature = "yoyo"))]
pub use self::banana::*;

//@has 'foo/struct.Yolo.html'
//@has - '//*[@class="item-info"]/*[@class="stab portability"]' 'Available on non-crate feature ananas and non-crate feature banana only.'
pub use self::banana::SubBanana as Yolo;

#[cfg(not(feature = "banana"))]
mod banana {
    /// Depends on `banana` feature.
    pub struct FlatBanana {}

    #[cfg(not(feature = "ananas"))]
    mod sub_banana {
        /// Also depends on `banana` feature.
        pub struct SubBanana {}
    }

    pub use self::sub_banana::*;
}
