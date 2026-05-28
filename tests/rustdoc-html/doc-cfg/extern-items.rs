// Ensure that the `cfg` on the extern blocks are correctly taken into account by
// their children.
// Regression test for <https://github.com/rust-lang/rust/issues/150268>.

#![feature(doc_cfg)]
#![crate_name = "foo"]

//@has 'foo/index.html'
//@count - '//*[@class="stab portability"]' 2
//@has - '//*[@class="stab portability"]' 'Non-banana'

//@has 'foo/fn.doc_cfg_doesnt_work.html'
//@has - '//*[@class="stab portability"]' 'Available on non-crate feature banana only.'

//@has 'foo/fn.doc_cfg_works.html'
//@has - '//*[@class="stab portability"]' 'Available on non-crate feature banana only.'

unsafe extern "C" {
    #[cfg(not(feature = "banana"))]
    pub fn doc_cfg_works();
}

#[cfg(not(feature = "banana"))]
unsafe extern "C" {
    pub fn doc_cfg_doesnt_work();
}
