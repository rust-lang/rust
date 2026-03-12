#![deny(self_lifetime_elision_not_applicable)]
#![allow(non_snake_case)]

use std::pin::Pin;

struct Foo<'a>(&'a ());
type Alias<'a> = Foo<'a>;

impl<'a> Foo<'a> {
    // &self / self: &Self -> no lint (genuine Self)
    fn ref_self(&self) -> &() {
        self.0
    }
    fn ref_Self(self: &Self) -> &() {
        self.0
    }
    fn box_ref_Self(self: Box<&Self>) -> &() {
        self.0
    }
    fn pin_ref_Self(self: Pin<&Self>) -> &() {
        self.0
    }

    // self: &Struct in impl Struct -> lint (name-match hack, not genuine Self)
    fn ref_Foo(self: &Foo<'a>) -> &() {
        self.0
    }
    //~^^^ ERROR `self` parameter type does not contain `Self`
    //~| WARN this was previously accepted by the compiler
    fn pin_ref_Foo(self: Pin<&Foo<'a>>) -> &() {
        self.0
    }
    //~^^^ ERROR `self` parameter type does not contain `Self`
    //~| WARN this was previously accepted by the compiler

    // self: &Alias in impl Struct -> lint + E0106 (hack misses alias, elision fails)
    fn ref_Alias_in_foo_impl(self: &Alias<'a>) -> &() {
        self.0
    }
    //~^^^ ERROR `self` parameter type does not contain `Self`
    //~| WARN this was previously accepted by the compiler
    //~| ERROR missing lifetime specifier [E0106]
}

impl<'a> Alias<'a> {
    // &self in impl Alias -> no lint (genuine Self)
    fn ref_self_in_alias_impl(&self) -> &() {
        self.0
    }

    // self: &Alias in impl Alias -> lint + E0106 (impl_self=None for TyAlias, elision fails)
    fn ref_Alias(self: &Alias<'a>) -> &() {
        self.0
    }
    //~^^^ ERROR `self` parameter type does not contain `Self`
    //~| WARN this was previously accepted by the compiler
    //~| ERROR missing lifetime specifier [E0106]

    // self: &Struct in impl Alias -> lint + E0106 (impl_self=None for TyAlias, elision fails)
    fn ref_Foo_in_alias_impl(self: &Foo<'a>) -> &() {
        self.0
    }
    //~^^^ ERROR `self` parameter type does not contain `Self`
    //~| WARN this was previously accepted by the compiler
    //~| ERROR missing lifetime specifier [E0106]
}

fn main() {}
