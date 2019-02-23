// Test a `exists<'a> { forall<'b> { 'a = 'b } }` pattern -- which should not compile!
//
// In particular, we test this pattern in trait solving, where it is not connected
// to any part of the source code.

use std::cell::Cell;

trait Trait<T> {}

fn foo<T>()
where
    T: Trait<for<'b> fn(Cell<&'b u32>)>,
{
}

impl<'a> Trait<fn(Cell<&'a u32>)> for () {}

fn main() {
    // Here, proving that `(): Trait<for<'b> fn(&'b u32)>` uses the impl:
    //
    // - The impl provides the clause `forall<'a> { (): Trait<fn(&'a u32)> }`
    // - We instantiate `'a` existentially to get `(): Trait<fn(&?a u32)>`
    // - We unify `fn(&?a u32)` with `for<'b> fn(&'b u32)`
    //   - This requires (among other things) instantiating `'b` universally,
    //     yielding `fn(&!b u32)`, in a fresh universe U1
    //   - So we get `?a = !b` but the universe U0 assigned to `?a` cannot name `!b`.

    foo::<()>(); //~ ERROR not satisfied
}
