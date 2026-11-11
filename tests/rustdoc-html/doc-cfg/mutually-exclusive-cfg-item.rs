// An item split into mutually-exclusive `#[cfg(..)]` variants that together cover
// every configuration must not get a portability note when `auto_cfg` is disabled.
// Regression test for <https://github.com/rust-lang/rust/issues/149786>.

#![feature(doc_cfg)]
#![crate_name = "foo"]

pub struct S;

impl S {
    //@ has 'foo/struct.S.html'
    //@ count - '//*[@id="method.new"]/..//*[@class="stab portability"]' 0
    #[doc(auto_cfg = false)]
    #[cfg(panic = "unwind")]
    pub fn new() -> S {
        S
    }

    #[doc(auto_cfg = false)]
    #[cfg(not(panic = "unwind"))]
    pub fn new() -> S {
        S
    }
}
