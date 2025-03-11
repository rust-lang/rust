//@ revisions: feat nofeat

#![cfg_attr(feat, feature(generic_const_parameter_types))]
//[feat]~^ WARN the feature `generic_const_parameter_types` is incomplete

trait Foo {
    type Assoc<const N: Self>;
    //~^ ERROR `Self` is forbidden as the type of a const generic parameter
}

fn foo<T: Foo>() {
    // We used to end up feeding the type of this anon const to be `T`, but the anon const
    // doesn't inherit the generics of `foo`, which led to index oob errors.
    let x: T::Assoc<3>;
    //~^ ERROR anonymous constants referencing generics are not yet supported
}

fn main() {}
