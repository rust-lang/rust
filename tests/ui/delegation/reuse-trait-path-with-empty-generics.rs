//@ needs-rustc-debug-assertions
#![feature(fn_delegation)]

trait Trait {
    fn bar4();
}

impl Trait for () {
    reuse Trait::<> as bar4;
    //~^ ERROR cannot find function `Trait` in this scope
}

fn main() {}
