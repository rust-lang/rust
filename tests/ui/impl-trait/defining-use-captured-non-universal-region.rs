// This was an ICE. See #110726.
// FIXME(aliemjay): outdated due to "once modulo regions" restriction.

//@ revisions: statik infer fixed
//@ [fixed] check-pass
#![allow(unconditional_recursion)]

fn foo<'a>() -> impl Sized + 'a {
    #[cfg(statik)]
    let i: i32 = foo::<'static>();
    //[statik]~^ ERROR opaque type used twice with different lifetimes
    //[statik]~| ERROR opaque type used twice with different lifetimes

    #[cfg(infer)]
    let i: i32 = foo::<'_>();
    //[infer]~^ ERROR expected generic lifetime parameter, found `'_`

    #[cfg(fixed)]
    let i: i32 = foo::<'a>();

    i
}

fn main() {}
