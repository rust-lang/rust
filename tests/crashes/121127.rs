//@ known-bug: #121127
//@ compile-flags: -Zpolymorphize=on -Zinline-mir=yes -C debuginfo=2
// Note that as of PR#123949 this only crashes with debuginfo enabled

#![feature(specialization)]

pub trait Foo {
    fn abc() -> u32;
}

pub trait Marker {}

impl<T> Foo for T {
    default fn abc(f: fn(&T), t: &T) -> u32 {
        16
    }
}

impl<T: Marker> Foo for T {
    fn def() -> u32 {
        Self::abc()
    }
}
