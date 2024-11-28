//@ known-bug: #121411
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn into_iter(&self) {}
}

impl const Foo for () {
    fn into_iter(a: u32, b: u32) {}
}

const _: () = Foo::into_iter(&());
