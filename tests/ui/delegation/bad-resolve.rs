#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    const C: u32 = 0;
    type Type;
    fn bar() {}
    fn foo(&self, x: i32) -> i32 { x }
}

struct F;
impl Trait for F {
    type Type = i32;
}

impl F {
    fn foo(&self, x: i32) -> i32 { x }
}

struct S(F);

impl Trait for S {
//~^ ERROR not all trait items implemented, missing: `Type`
    reuse <F as Trait>::C;
    //~^ ERROR item `C` is an associated method, which doesn't match its trait `Trait`
    //~| ERROR expected function, found associated constant `Trait::C`
    reuse <F as Trait>::Type;
    //~^ ERROR item `Type` is an associated method, which doesn't match its trait `Trait`
    //~| ERROR expected method or associated constant, found associated type `Trait::Type`
    reuse <F as Trait>::baz;
    //~^ ERROR method `baz` is not a member of trait `Trait`
    //~| ERROR cannot find method or associated constant `baz` in trait `Trait`
    reuse <F as Trait>::bar;

    reuse foo { &self.0 }
    //~^ ERROR cannot find function `foo` in this scope
    reuse Trait::foo2 { self.0 }
    //~^ ERROR cannot find function `foo2` in trait `Trait`
    //~| ERROR method `foo2` is not a member of trait `Trait`
}

mod prefix {}
reuse unresolved_prefix::{a, b, c}; //~ ERROR use of unresolved module or unlinked crate
reuse prefix::{self, super, crate}; //~ ERROR `crate` in paths can only be used in start position

fn main() {}
