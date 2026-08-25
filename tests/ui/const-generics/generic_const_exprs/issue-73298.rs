//@ build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

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
    //type Baz = [u8; 1];
}

fn main() {
    <() as Foo>::bar();
}
