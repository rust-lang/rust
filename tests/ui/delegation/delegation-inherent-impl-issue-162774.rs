#![feature(fn_delegation)]

trait Trait {
    fn foo(&self) {}
}

struct F;
impl Trait for F {}

struct S(F);

impl S {
    reuse Trait::foo { self.0 }
    reuse Self::foo::<> as bar { self }
    //~^ ERROR failed to resolve delegation to inherent impl
}

fn main() {}
