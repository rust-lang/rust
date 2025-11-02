// Regression test for issue #142488, a diagnostics ICE when trying to suggest missing methods
// present in similar tuple types.
// This is a few of the MCVEs from the issues and its many duplicates.

// 1
fn main() {
    for a in x {
        //~^ ERROR: cannot find value `x` in this scope
        (a,).to_string()
        //~^ ERROR: the method `to_string` exists for tuple
    }
}

// 2
trait Trait {
    fn meth(self);
}

impl<T, U: Trait> Trait for (T, U) {
    fn meth(self) {}
}

fn mcve2() {
    ((), std::collections::HashMap::new()).meth()
    //~^ ERROR: the method `meth` exists for tuple
}

// 3
trait I {}

struct Struct;
impl I for Struct {}

trait Tr {
    fn f<A>(self) -> (A,)
    where
        Self: Sized,
    {
        loop {}
    }
}

impl<T> Tr for T where T: I {}

fn mcve3() {
    Struct.f().f();
    //~^ ERROR: the method `f` exists for tuple
}
