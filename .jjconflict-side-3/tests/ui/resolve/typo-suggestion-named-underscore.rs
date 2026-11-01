const _: () = ();

fn main() {
    a // Shouldn't suggest underscore
    //~^ ERROR: cannot find value `a` in this scope
}

trait Unknown {}

#[allow(unused_imports)]
use Unknown as _;

fn foo<T: A>(x: T) {} // Shouldn't suggest underscore
//~^ ERROR: cannot find trait `A` in this scope
