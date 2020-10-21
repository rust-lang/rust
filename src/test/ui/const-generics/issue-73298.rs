// build-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

use std::convert::AsMut;
use std::default::Default;

trait Foo: Sized {
    type Baz: Default + AsMut<[u8]>;
    fn bar() {
        Self::Baz::default().as_mut();
    }
}

impl Foo for () {
    type Baz = [u8; 1 * 1];
}

fn main() {
    <() as Foo>::bar();
}
