//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

pub trait Foo {
    #[cfg_attr(true, type_const)] //~ ERROR `type_const` within an `#[cfg_attr]` attribute is forbidden
    const N: usize;
}

impl Foo for u32 {
    #[cfg_attr(true, type_const)] //~ ERROR `type_const` within an `#[cfg_attr]` attribute is forbidden
    const N: usize = 99;
}

fn main() {}
