#![feature(const_trait_impl)]
// known-bug: #110395
// revisions: yy yn ny nn

#[cfg_attr(any(yy, yn), const_trait)]
trait Foo {
    fn a(&self);
}

#[cfg_attr(any(yy, ny), const_trait)]
trait Bar: ~const Foo {}
// FIXME [ny,nn]~^ ERROR: ~const can only be applied to `#[const_trait]`
// FIXME [ny,nn]~| ERROR: ~const can only be applied to `#[const_trait]`

const fn foo<T: Bar>(x: &T) {
    x.a();
    // FIXME [yn,yy]~^ ERROR the trait bound
}

fn main() {}
