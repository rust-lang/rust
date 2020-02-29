// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash
#![feature(lazy_normalization_consts)]
//~^ WARN the feature `lazy_normalization_consts` is incomplete and may cause the compiler to crash

struct Const<const N: usize>;
trait Foo<const N: usize> {}

impl<const N: usize> Foo<{N}> for Const<{N}> {}

fn foo_impl(_: impl Foo<3>) {}

fn foo_explicit<T: Foo<3>>(_: T) {}

fn foo_where<T>(_: T) where T: Foo<3> {}

fn main() {
    // FIXME this causes a stack overflow in rustc
    // foo_impl(Const);
    foo_impl(Const::<3>);

    // FIXME this causes a stack overflow in rustc
    // foo_explicit(Const);
    foo_explicit(Const::<3>);

    // FIXME this causes a stack overflow in rustc
    // foo_where(Const);
    foo_where(Const::<3>);
}
