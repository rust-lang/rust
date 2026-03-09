#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait{
    fn bar();
}

impl Trait for () {
    reuse missing::<> as bar;
    //~^ ERROR: cannot find function `missing` in this scope
}

fn main() {}
