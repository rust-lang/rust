#![crate_type = "lib"]
#![feature(rustdoc_internals)]

#![doc(keyword = "match")]
//~^ ERROR `#![doc(keyword = "...")]` isn't allowed as a crate-level attribute
#[doc(keyword = "match")] //~ ERROR `#[doc(keyword = "...")]` should be used on empty modules
mod foo {
    fn hell() {}
}

#[doc(keyword = "match")] //~ ERROR `#[doc(keyword = "...")]` should be used on modules
fn foo() {}


// Regression test for the ICE described in #83512.
trait Foo {
    #[doc(keyword = "match")]
    //~^ ERROR: `#[doc(keyword = "...")]` should be used on modules
    fn quux() {}
}

#[doc(keyword = "tadam")] //~ ERROR nonexistent keyword `tadam`
mod tadam {}
