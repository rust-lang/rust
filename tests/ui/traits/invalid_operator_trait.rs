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

    fn add<Y>(self, _: RHS) -> Self::Output;
    //~^ ERROR `add` must not have any generic parameters
}

#[allow(unreachable_code)]
fn ice(a: usize) {
    let r = loop {};
    r = r + a;
}
