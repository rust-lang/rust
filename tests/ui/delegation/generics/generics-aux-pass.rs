//@ aux-crate:generics=generics.rs
//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

reuse generics::foo as bar;
reuse generics::Trait::foo as trait_foo;

reuse generics::foo::<'static, 'static, i32, i32, 1> as bar1;
reuse generics::Trait::<'static, i32, 1>::foo::<'static, i32, false> as trait_foo1;

#[derive(Clone, Copy)]
struct X;

impl generics::Trait<'static, i32, 1> for X {}

impl X {
    reuse generics::foo as bar;
    reuse generics::Trait::foo as trait_foo;

    reuse generics::foo::<'static, 'static, i32, i32, 1> as bar1;
    reuse generics::Trait::<'static, i32, 1>::foo::<'static, i32, false> as trait_foo1;
}

trait LocalTrait {
    fn get() -> u8 { 123 }
    fn get_self(&self) -> u8 { 123 }

    reuse generics::foo as bar;
    reuse generics::foo::<'static, 'static, i32, i32, 1> as bar1;

    reuse generics::Trait::foo as trait_foo { Self::get() }
    reuse generics::Trait::<'static, i32, 1>::foo::<'static, i32, false> as trait_foo1 {
        Self::get_self(&self)
    }
}

impl LocalTrait for usize {}

fn main() {
    bar::<i32, i32, 1>();
    bar::<'static, 'static, i32, i32, 1>();
    trait_foo::<'static, 'static, u8, i32, 1, String, true>(123);

    bar1();
    trait_foo1::<u8>(123);

    let x = X{};

    X::bar::<i32, i32, 1>();
    X::bar::<'static, 'static, i32, i32, 1>();
    X::bar1();
    x.trait_foo::<'static, 'static, i32, 1, String, true>();
    x.trait_foo1();

    <usize as LocalTrait>::bar::<i32, i32, 1>();
    <usize as LocalTrait>::bar::<'static, 'static, i32, i32, 1>();
    <usize as LocalTrait>::bar1();

    1usize.trait_foo::<'static, 'static, i32, 1, String, true>();
    1usize.trait_foo1();
}
