//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args)]
#![feature(min_generic_const_args)]

trait Trait {
    const F: fn();
}

fn take(_: impl Trait<F = { || {} }>) {}
//~^ ERROR

type TraitObject = dyn Trait<F = { || {} }>;
//~^ ERROR

fn take_trait_object(_: &dyn Trait<F = { || {} }>) {}
//~^ ERROR

fn local_trait_object() {
    let _: &dyn Trait<F = { || {} }>;
    //~^ ERROR
}

struct Impl;

fn function() {}

impl Trait for Impl {
    const F: fn() = function;
}

fn coerce_trait_object(value: &Impl) {
    let _: &dyn Trait<F = { || {} }> = value;
    //~^ ERROR
}

enum Foo {
    Unit,
    Function(fn()),
}

trait EnumTrait {
    const F: Foo;
}

fn take_unit(_: impl EnumTrait<F = { Foo::Unit }>) {}

fn take_function(_: impl EnumTrait<F = { Foo::Function(|| {}) }>) {}
//~^ ERROR

fn main() {}
