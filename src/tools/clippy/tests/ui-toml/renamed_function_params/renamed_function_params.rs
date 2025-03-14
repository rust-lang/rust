//@no-rustfix
//@revisions: default extend
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/renamed_function_params/default
//@[extend] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/renamed_function_params/extend
#![warn(clippy::renamed_function_params)]
#![allow(clippy::partialeq_ne_impl, clippy::to_string_trait_impl)]
#![allow(unused)]

use std::hash::{Hash, Hasher};

struct A;
impl From<A> for String {
    fn from(_value: A) -> Self {
        String::new()
    }
}
impl ToString for A {
    fn to_string(&self) -> String {
        String::new()
    }
}

struct B(u32);
impl std::convert::From<B> for String {
    fn from(b: B) -> Self {
        b.0.to_string()
    }
}
impl PartialEq for B {
    fn eq(&self, rhs: &Self) -> bool {
        //~^ renamed_function_params
        self.0 == rhs.0
    }
    fn ne(&self, rhs: &Self) -> bool {
        //~^ renamed_function_params
        self.0 != rhs.0
    }
}

trait MyTrait {
    fn foo(&self, val: u8);
    fn bar(a: u8, b: u8);
    fn baz(self, _val: u8);
    fn quz(&self, _: u8);
}

impl MyTrait for B {
    fn foo(&self, i_dont_wanna_use_your_name: u8) {} // only lint in `extend`
    //
    //~[default]^^ renamed_function_params
    fn bar(_a: u8, _: u8) {}
    fn baz(self, val: u8) {}
    fn quz(&self, val: u8) {}
}

impl Hash for B {
    fn hash<H: Hasher>(&self, states: &mut H) {
        //~^ renamed_function_params
        self.0.hash(states);
    }
    fn hash_slice<H: Hasher>(date: &[Self], states: &mut H) {
        //~^ renamed_function_params
        for d in date {
            d.hash(states);
        }
    }
}

impl B {
    fn totally_irrelevant(&self, right: bool) {}
    fn some_fn(&self, other: impl MyTrait) {}
}

#[derive(Copy, Clone)]
enum C {
    A,
    B(u32),
}

impl std::ops::Add<B> for C {
    type Output = C;
    fn add(self, b: B) -> C {
        //~[default]^ renamed_function_params
        // only lint in `extend`
        C::B(b.0)
    }
}

impl From<A> for C {
    fn from(_: A) -> C {
        C::A
    }
}

trait CustomTraitA {
    fn foo(&self, other: u32);
}
trait CustomTraitB {
    fn bar(&self, value: u8);
}

macro_rules! impl_trait {
    ($impl_for:ident, $tr:ty, $fn_name:ident, $t:ty) => {
        impl $tr for $impl_for {
            fn $fn_name(&self, v: $t) {}
        }
    };
}

impl_trait!(C, CustomTraitA, foo, u32);
impl_trait!(C, CustomTraitB, bar, u8);

fn main() {}
