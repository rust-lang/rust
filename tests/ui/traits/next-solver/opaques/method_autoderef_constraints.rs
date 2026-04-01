//@ compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative/issues/263
// Previously `method_auto_deref_steps` would also return opaque
// types which have already been defined in the parent context.
//
// We then handled these opaque types by emitting `AliasRelate` goals
// when instantiating its result, assuming that operation to be infallible.
// By returning opaque type constraints from the parent context and
// constraining the hidden type without reproving the item bounds of
// the opaque, this ended up causing ICE.

use std::ops::Deref;
trait Trait {}
struct Inv<T>(*mut T);
impl Trait for i32 {}
impl Deref for Inv<u32> {
    type Target = u32;
    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

fn mk<T>() -> T { todo!() }
fn foo() -> Inv<impl Trait> {
    //~^ ERROR: the trait bound `u32: Trait` is not satisfied [E0277]
    let mut x: Inv<_> = mk();
    if false {
        return x;
        //~^ ERROR: the trait bound `u32: Trait` is not satisfied [E0277]
    }

    x.count_ones();
    x
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
