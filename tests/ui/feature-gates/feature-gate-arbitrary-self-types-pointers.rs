//@ revisions: default feature
#![cfg_attr(feature, feature(arbitrary_self_types))]

trait Foo {
    fn foo(self: *const Self); //~ ERROR `*const Self` cannot be used as the type of `self`
}

struct Bar;

impl Foo for Bar {
    fn foo(self: *const Self) {} //~ ERROR `*const Bar` cannot be used as the type of `self`
}

impl Bar {
    fn bar(self: *mut Self) {} //~ ERROR `*mut Bar` cannot be used as the type of `self`
}

fn main() {}
