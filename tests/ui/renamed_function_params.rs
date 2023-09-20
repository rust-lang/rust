#![warn(clippy::renamed_function_params)]
#![allow(clippy::partialeq_ne_impl)]
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
impl From<B> for String {
    fn from(b: B) -> Self {
        //~^ ERROR: function parameter name was renamed from its trait default
        b.0.to_string()
    }
}
impl PartialEq for B {
    fn eq(&self, rhs: &Self) -> bool {
        //~^ ERROR: function parameter name was renamed from its trait default
        self.0 == rhs.0
    }
    fn ne(&self, rhs: &Self) -> bool {
        //~^ ERROR: function parameter name was renamed from its trait default
        self.0 != rhs.0
    }
}

trait MyTrait {
    fn foo(&self, val: u8);
    fn bar(a: u8, b: u8);
    fn baz(self, _val: u8);
}

impl MyTrait for B {
    fn foo(&self, i_dont_wanna_use_your_name: u8) {}
    //~^ ERROR: function parameter name was renamed from its trait default
    fn bar(_a: u8, _b: u8) {}
    fn baz(self, val: u8) {}
}

impl Hash for B {
    fn hash<H: Hasher>(&self, states: &mut H) {
        //~^ ERROR: function parameter name was renamed from its trait default
        self.0.hash(states);
    }
    fn hash_slice<H: Hasher>(date: &[Self], states: &mut H) {
        //~^ ERROR: function parameter name was renamed from its trait default
        for d in date {
            d.hash(states);
        }
    }
}

impl B {
    fn totally_irrelevant(&self, right: bool) {}
    fn some_fn(&self, other: impl MyTrait) {}
}

fn main() {}
