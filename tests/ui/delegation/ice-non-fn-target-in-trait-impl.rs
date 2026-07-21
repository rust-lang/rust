// Regression test for #153743 and #153744.
// Delegation to a module or crate root inside a trait impl
// should emit a resolution error, not ICE.

#![feature(fn_delegation)]

trait Trait {
    fn bar();
    fn bar2();
}

impl Trait for () {
    reuse std::path::<> as bar;
    //~^ ERROR cannot find function `path` in crate `std`
    reuse core::<> as bar2;
    //~^ ERROR cannot find function `core` in this scope
}

fn main() {}
