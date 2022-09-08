// Test a case where variance and higher-ranked types interact in surprising ways.
//
// In particular, we test this pattern in trait solving, where it is not connected
// to any part of the source code.
//
// check-pass

trait Trait<T> {}

fn foo<T>()
where
    T: Trait<for<'b> fn(fn(&'b u32))>,
{
}

impl<'a> Trait<fn(fn(&'a u32))> for () {}

fn main() {
    // Here, proving that `(): Trait<for<'b> fn(&'b u32)>` uses the impl:
    //
    // - The impl provides the clause `forall<'a> { (): Trait<fn(fn(&'a u32))> }`
    // - We instantiate `'a` existentially to get `(): Trait<fn(fn(&?a u32))>`
    // - We unify `fn(fn(&?a u32))` with `for<'b> fn(fn(&'b u32))` -- this does a
    //   "bidirectional" subtyping check, so we wind up with:
    //   - `fn(fn(&?a u32)) <: for<'b> fn(fn(&'b u32))` :-
    //     - `fn(&!b u32) <: fn(&?a u32)`
    //       - `&?a u32 <: &!b u32`
    //         - `?a: !'b` -- solveable if `?a` is inferred to `'static`
    //   - `for<'b> fn(fn(&'b u32)) <: fn(fn(&?a u32))` :-
    //     - `fn(&?a u32) <: fn(&?b u32)`
    //       - `&?b u32 <: &?a u32`
    //         - `?b: ?a` -- solveable if `?b` is inferred to `'static`
    // - So the subtyping check succeeds, somewhat surprisingly.
    //   This is because we can use `'static`.

    foo::<()>();
}
