// This test ensures that paths with generics still get their link to their definition
// correctly generated.

//@ compile-flags: -Zunstable-options --generate-link-to-definition
#![crate_name = "foo"]

//@ has 'src/foo/link-on-path-with-generics.rs.html'

pub struct Soyo<T>(T);
pub struct Saya;

//@ has - '//pre[@class="rust"]//a[@href="#9"]' 'Soyo'
//@ has - '//pre[@class="rust"]//a[@href="#10"]' 'Saya'
pub fn bar<T>(s: Soyo<T>, x: Saya) {}
