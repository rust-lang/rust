// compile-flags: -Zborrowck=mir

#![allow(dead_code)]

// Test that we relate the type of the fn type to the type of the fn
// ptr when doing a `ReifyFnPointer` cast.
//
// This test is a bit tortured, let me explain:
//

// The `where 'a: 'a` clause here ensures that `'a` is early bound,
// which is needed below to ensure that this test hits the path we are
// concerned with.
fn foo<'a>(x: &'a u32) -> &'a u32
where
    'a: 'a,
{
    panic!()
}

fn bar<'a>(x: &'a u32) -> &'static u32 {
    // Here, the type of `foo` is `typeof(foo::<'x>)` for some fresh variable `'x`.
    // During NLL region analysis, this will get renumbered to `typeof(foo::<'?0>)`
    // where `'?0` is a new region variable.
    //
    // (Note that if `'a` on `foo` were late-bound, the type would be
    // `typeof(foo)`, which would interact differently with because
    // the renumbering later.)
    //
    // This type is then coerced to a fn type `fn(&'?1 u32) -> &'?2
    // u32`. Here, the `'?1` and `'?2` will have been created during
    // the NLL region renumbering.
    //
    // The MIR type checker must therefore relate `'?0` to `'?1` and `'?2`
    // as part of checking the `ReifyFnPointer`.
    let f: fn(_) -> _ = foo;
    f(x)
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
