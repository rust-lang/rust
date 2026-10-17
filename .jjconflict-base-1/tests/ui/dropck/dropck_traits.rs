//@ run-pass
//! Regression test for #34426, regarding HRTB in drop impls

// All of this Drop impls should compile.

pub trait Lifetime<'a> {}
impl<'a> Lifetime<'a> for i32 {}

#[allow(dead_code)]
struct Foo<L>
where
    for<'a> L: Lifetime<'a>,
{
    l: L,
}

impl<L> Drop for Foo<L>
where
    for<'a> L: Lifetime<'a>,
{
    fn drop(&mut self) {}
}

#[allow(dead_code)]
struct Foo2<L>
where
    for<'a> L: Lifetime<'a>,
{
    l: L,
}

impl<T: for<'a> Lifetime<'a>> Drop for Foo2<T>
where
    for<'x> T: Lifetime<'x>,
{
    fn drop(&mut self) {}
}

pub trait Lifetime2<'a, 'b> {}
impl<'a, 'b> Lifetime2<'a, 'b> for i32 {}

#[allow(dead_code)]
struct Bar<L>
where
    for<'a, 'b> L: Lifetime2<'a, 'b>,
{
    l: L,
}

impl<L> Drop for Bar<L>
where
    for<'a, 'b> L: Lifetime2<'a, 'b>,
{
    fn drop(&mut self) {}
}

#[allow(dead_code)]
struct FnHolder<T: for<'a> Fn(&'a T, dyn for<'b> Lifetime2<'a, 'b>) -> u8>(T);

impl<T: for<'a> Fn(&'a T, dyn for<'b> Lifetime2<'a, 'b>) -> u8> Drop for FnHolder<T> {
    fn drop(&mut self) {}
}

fn main() {
    let _foo = Foo { l: 0 };

    let _bar = Bar { l: 0 };
}
