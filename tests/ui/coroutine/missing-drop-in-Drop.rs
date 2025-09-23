// issue #122630: ICE Assignment between coroutine saved locals whose storage is not
// marked as conflicting
//@ compile-flags: -Zvalidate-mir

use std::ops::Coroutine;
//~^ ERROR use of unstable library feature `coroutine_trait`

const FOO_SIZE: usize = 1024;
struct Foo([u8; FOO_SIZE]);

impl Drop for Foo {
    //~^ ERROR not all trait items implemented, missing: `drop`
    fn move_before_yield_with_noop() -> impl Coroutine<Yield = ()> {}
    //~^ ERROR method `move_before_yield_with_noop` is not a member of trait `Drop`
    //~| ERROR use of unstable library feature `coroutine_trait`
    //~| ERROR use of unstable library feature `coroutine_trait`
    //~| ERROR use of unstable library feature `coroutine_trait`
    //~| ERROR the trait bound `(): Coroutine` is not satisfied
}

fn overlap_move_points() -> impl Coroutine<Yield = ()> {
    //~^ ERROR use of unstable library feature `coroutine_trait`
    //~| ERROR use of unstable library feature `coroutine_trait`
    //~| ERROR use of unstable library feature `coroutine_trait`
    static || {
        //~^ ERROR coroutine syntax is experimental
        let first = Foo([0; FOO_SIZE]);
        yield;
        //~^ ERROR yield syntax is experimental
        //~| ERROR yield syntax is experimental
        //~| ERROR `yield` can only be used in `#[coroutine]` closures, or `gen` blocks
        let second = first;
        yield;
        //~^ ERROR yield syntax is experimental
        //~| ERROR yield syntax is experimental
        let second = first; //~ ERROR use of moved value: `first`
        yield;
        //~^ ERROR yield syntax is experimental
        //~| ERROR yield syntax is experimental
    }
}
//~^ ERROR `main` function not found
