// Test a `exists<'a> { forall<'b> { 'a = 'b } }` pattern -- which should not compile!
//
// In particular, we test this pattern in trait solving, where it is not connected
// to any part of the source code.

trait Trait<T> {}

fn foo<'a>() -> fn(&'a u32) {
    panic!()
}

fn main() {
    // Here, proving that `(): Trait<for<'b> fn(&'b u32)>` uses the impl:
    //
    // - The impl provides the clause `forall<'a> { (): Trait<fn(&'a u32)> }`
    // - We instantiate `'a` existentially to get `(): Trait<fn(&?a u32)>`
    // - We unify `fn(&?a u32)` with `for<'b> fn(&'b u32)`
    //   - This requires (among other things) instantiating `'b` universally,
    //     yielding `fn(&!b u32)`, in a fresh universe U1
    //   - So we get `?a = !b` but the universe U0 assigned to `?a` cannot name `!b`.

    let _: for<'b> fn(&'b u32) = foo(); //~ ERROR cannot infer
}
