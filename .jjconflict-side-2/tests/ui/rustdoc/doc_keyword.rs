#![crate_type = "lib"]
#![feature(rustdoc_internals)]

#![doc(keyword = "match")]
//~^ ERROR `#![doc(keyword = "...")]` isn't allowed as a crate-level attribute

#[doc(keyword = "match")] //~ ERROR `#[doc(keyword = "...")]` should be used on anonymous constants
fn foo() {}


// Regression test for the ICE described in #83512.
trait Foo {
    #[doc(keyword = "match")]
    //~^ ERROR: `#[doc(keyword = "...")]` should be used on anonymous constants
    fn quux() {}
}

#[doc(keyword = "tadam")] //~ ERROR nonexistent keyword `tadam`
const _: () = ();
