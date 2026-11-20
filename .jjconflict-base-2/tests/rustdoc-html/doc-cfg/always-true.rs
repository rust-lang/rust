// Some generated "cfg text" check.
// Regression test for <https://github.com/rust-lang/rust/issues/145075>.

#![feature(doc_cfg)]
#![crate_name = "foo"]

// This one doesn't display any cfg label.
//@ has 'foo/fn.f.html'
//@ count - '//*[@class="stab portability"]' 0
#[cfg(any(all()))]
pub fn f() {}

//@ has 'foo/fn.f2.html'
// If you change the selector in this one, don't forget to update the one of the `f`
// function as well!
//@ count - '//*[@class="stab portability"]' 1
//@ has - '//*[@class="stab portability"]' 'Available everywhere.'
// This one display a cfg label.
#[cfg(any(true))]
pub fn f2() {}
