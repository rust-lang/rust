// Test that we do not ICE when the self type is `ty::err`, but rather
// just propagate the error.

#![crate_type = "lib"]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

#[lang="sized"]
pub trait Sized {
    // Empty.
}

#[lang = "add"]
trait Add<RHS=Self> {
    type Output;

    fn add(self, _: RHS) -> Self::Output;
}

fn ice<A>(a: A) {
    let r = loop {};
    r = r + a;
    //~^ ERROR the trait bound `(): Add<A>` is not satisfied
}
