//@ needs-rustc-debug-assertions
#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn bar4();
}

impl Trait for () {
    reuse Trait::<> as bar4;
    //~^ ERROR expected function, found trait `Trait`
}

fn main() {}
