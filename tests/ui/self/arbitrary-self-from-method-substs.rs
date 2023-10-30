// revisions: default feature
#![cfg_attr(feature, feature(arbitrary_self_types))]

use std::ops::Deref;

struct Foo(u32);
impl Foo {
    fn get<R: Deref<Target = Self>>(self: R) -> u32 {
        //[default]~^ ERROR: `R` cannot be used as the type of `self`
        self.0
    }
}

fn main() {
    let mut foo = Foo(1);
    foo.get::<&Foo>();
    //[feature]~^ ERROR mismatched types
}
