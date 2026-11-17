//! Regression test for <https://github.com/rust-lang/rust/issues/162337>.
//@ build-pass
//@ compile-flags: -Znext-solver=globally
#![feature(gca_min_const_items, gca_const_items)]

use std::marker::PhantomData;

struct Meta<A> {
    value: i32,
    type_: PhantomData<A>,
}

trait MetaTrait {
    fn get_value(&self) -> i32;
}

impl<A> MetaTrait for Meta<A> {
    fn get_value(&self) -> i32 {
        self.value
    }
}

impl MetaTrait for () {
    fn get_value(&self) -> i32 {
        0
    }
}

trait Bar {
    fn get_const(&self) -> &dyn MetaTrait;
}

struct Foo<A> {
    _value: A,
}

// An inherent const, as in the original report.
impl<A: 'static> Foo<A> {
    const CONST: &'static dyn MetaTrait = &Meta::<Self> { value: 0, type_: PhantomData };
}

impl<A: 'static> Bar for Foo<A> {
    fn get_const(&self) -> &dyn MetaTrait {
        Self::CONST
    }
}

// A trait const, which kept crashing after the inherent case stopped doing so.
trait HasConst {
    const CONST: &'static dyn MetaTrait;
}

impl<A: 'static> HasConst for Option<A> {
    const CONST: &'static dyn MetaTrait = &();
}

fn get<T: HasConst>() -> &'static dyn MetaTrait {
    T::CONST
}

fn main() {
    let foo = Foo { _value: 0 };
    let _bar: &dyn Bar = &foo;
    let _ = get::<Option<u8>>();
}
