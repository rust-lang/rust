//! The same as the non-ICE test, but const eval will run typeck of
//! `get` before running wfcheck (as that may in itself trigger const
//! eval again, and thus cause bogus cycles). This used to ICE because
//! we asserted that an error had already been emitted.

use std::ops::Deref;

struct Foo(u32);
impl Foo {
    const fn get<R: Deref<Target = Self>>(self: R) -> u32 {
        //~^ ERROR: `R` cannot be used as the type of `self`
        //~| ERROR destructor of `R` cannot be evaluated at compile-time
        self.0
        //~^ ERROR cannot call non-const fn `<R as Deref>::deref` in constant function
    }
}

const FOO: () = {
    let foo = Foo(1);
    foo.get::<&Foo>();
};

const BAR: [(); {
    FOO;
    0
}] = [];

fn main() {}
