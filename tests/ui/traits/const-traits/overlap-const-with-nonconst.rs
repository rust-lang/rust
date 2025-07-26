//@ revisions: spec min_spec

#![feature(const_trait_impl)]
#![cfg_attr(spec, feature(specialization))]
//[spec]~^ WARN the feature `specialization` is incomplete
#![cfg_attr(min_spec, feature(min_specialization))]

#[const_trait]
trait Bar {}
impl<T> const Bar for T {}

#[const_trait]
trait Foo {
    fn method(&self);
}
impl<T> const Foo for T
where
    T: [const] Bar,
{
    default fn method(&self) {}
}
// specializing impl:
impl<T> Foo for (T,) {
//~^ ERROR conflicting implementations
    fn method(&self) {
        println!("hi");
    }
}

const fn dispatch<T: [const] Bar + Copy>(t: T) {
    t.method();
}

fn main() {
    const {
        dispatch(((),));
    }
}
