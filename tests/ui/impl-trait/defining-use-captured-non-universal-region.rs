// This was an ICE. See #110726.

//@ revisions: statik infer fixed next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [fixed] check-pass
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![allow(unconditional_recursion)]

fn foo<'a>() -> impl Sized + 'a {
    #[cfg(any(statik, next))]
    let i: i32 = foo::<'static>();
    //[statik]~^ ERROR expected generic lifetime parameter, found `'static`

    #[cfg(any(infer, next))]
    let i: i32 = foo::<'_>();
    //[infer]~^ ERROR expected generic lifetime parameter, found `'_`

    #[cfg(fixed)]
    let i: i32 = foo::<'a>();

    i
}

fn main() {}
