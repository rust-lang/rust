//@ aux-build: unstable_same_trait.rs
//@ revisions: disabled enabled
//@[disabled] run-pass
//@[enabled] run-pass

#![cfg_attr(enabled, feature(unstable_value))]
#![allow(dead_code)]

extern crate unstable_same_trait;
use unstable_same_trait::Unstable;

trait Stable<T> {
    fn value(&self) -> T;
}
impl<T> Stable<u8> for T {
    fn value(&self) -> u8 {
        99
    }
}
impl<T> Stable<u16> for T {
    fn value(&self) -> u16 {
        199
    }
}

fn call<T: Unstable<u8> + Unstable<u16>>(x: &T) -> u8 {
    // T implements Stable<T>, Unstable<u8> and Unstable<u16> here. We need to
    // ensure that we filter out unstable methods from bounds in the first pass
    // which checks traits from bounds, before the second pass which looks at
    // traits in scope.
    x.value()
    //[disabled]~^ WARN a method with this name may be added
    //[disabled]~| WARN once this associated item is added
}

fn main() {
    #[cfg(disabled)]
    assert_eq!(call(&()), 99);
    #[cfg(enabled)]
    assert_eq!(call(&()), 8);
}
