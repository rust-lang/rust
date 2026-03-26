// Regression test for #153743 and #153744.
// Delegation to a module or crate root inside a trait impl
// should emit a resolution error, not ICE.

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn bar();
    fn bar2();
}

impl Trait for () {
    reuse std::path::<> as bar;
    //~^ ERROR expected function, found module `std::path`
    reuse core::<> as bar2;
    //~^ ERROR expected function, found crate `core`
}

fn main() {}
