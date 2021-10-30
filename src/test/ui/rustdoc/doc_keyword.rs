#![crate_type = "lib"]
#![feature(rustdoc_internals)]

#![doc(keyword = "hello")] //~ ERROR

#[doc(keyword = "hell")] //~ ERROR
mod foo {
    fn hell() {}
}

#[doc(keyword = "hall")] //~ ERROR
fn foo() {}


// Regression test for the ICE described in #83512.
trait Foo {
    #[doc(keyword = "match")]
    //~^ ERROR: `#[doc(keyword = "...")]` can only be used on modules
    fn quux() {}
}
