// compile-flags: -Ztrait-solver=next

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

// Do not treat the RHS of a projection-goal as an unconstrained `Certainty::Yes` response
// if the impl is still further specializable.

trait Default {
   type Id;
}

impl<T> Default for T {
   default type Id = T; //~ ERROR type annotations needed
}

fn test<T: Default<Id = U>, U>() {}

fn main() {
    test::<u32, ()>();
    //~^ ERROR cannot satisfy `<u32 as Default>::Id == ()`
}
