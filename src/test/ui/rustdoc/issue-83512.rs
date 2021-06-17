// Regression test for the ICE described in #83512.

#![feature(doc_keyword)]
#![crate_type="lib"]

trait Foo {
    #[doc(keyword = "match")]
    //~^ ERROR: `#[doc(keyword = "...")]` can only be used on modules
    fn quux() {}
}
