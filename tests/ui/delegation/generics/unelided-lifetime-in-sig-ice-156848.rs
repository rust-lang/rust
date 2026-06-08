//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]

trait Trait {
    fn foo<'a: 'a>(&self) {}
}

struct F;
impl Trait for F {}

struct S(F);
impl S {
    reuse Trait::foo::<> { self.0 }
    //~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
    reuse Trait::foo::<'_> as bar { self.0 }
    //~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
    //~| WARN: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
