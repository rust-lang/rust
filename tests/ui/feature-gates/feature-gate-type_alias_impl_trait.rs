use std::fmt::Debug;

type Foo = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable

struct Bar(Foo);
#[define_opaque(Foo)] //~ ERROR use of unstable library feature `type_alias_impl_trait`
fn define() -> Bar {
    Bar(42)
}

type Foo2 = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable

#[define_opaque(Foo2)] //~ ERROR use of unstable library feature `type_alias_impl_trait`
fn define2() {
    let x = || -> Foo2 { 42 };
}

type Foo3 = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable

#[define_opaque(Foo3)] //~ ERROR use of unstable library feature `type_alias_impl_trait`
fn define3(x: Foo3) {
    let y: i32 = x;
}
#[define_opaque(Foo3)] //~ ERROR use of unstable library feature `type_alias_impl_trait`
fn define3_1() {
    define3(42)
}

type Foo4 = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable

#[define_opaque(Foo4)] //~ ERROR use of unstable library feature `type_alias_impl_trait`
fn define4(_: Foo4) {
    let y: Foo4 = 42;
}

type Foo5 = [(); {
    type Foo = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable
    //~^ ERROR unconstrained opaque type
    0
}];

fn main() {}
