//@ check-pass
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

    reuse Trait::foo::<'_> as bar { self.0 }
}

fn main() {}
