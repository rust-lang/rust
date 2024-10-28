// This test ensures that private/hidden items don't create ambiguity.
// This is a regression test for <https://github.com/rust-lang/rust/issues/130233>.

#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "foo"]

pub struct Thing {}

#[doc(hidden)]
#[allow(non_snake_case)]
pub fn Thing() {}

pub struct Bar {}

#[allow(non_snake_case)]
fn Bar() {}

//@ has 'foo/fn.repro.html'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]//a/@href' 'struct.Thing.html'
/// Do stuff with [`Thing`].
pub fn repro(_: Thing) {}

//@ has 'foo/fn.repro2.html'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]//a/@href' 'struct.Bar.html'
/// Do stuff with [`Bar`].
pub fn repro2(_: Bar) {}
