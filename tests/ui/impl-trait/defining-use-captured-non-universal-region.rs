// This was an ICE. See #110726.

//@ revisions: statik infer fixed
//@ [fixed] check-pass
#![allow(unconditional_recursion)]

fn foo<'a>() -> impl Sized + 'a {
    #[cfg(statik)]
    let i: i32 = foo::<'static>();
    //[statik]~^ ERROR expected generic lifetime parameter, found `'static`

    #[cfg(infer)]
    let i: i32 = foo::<'_>();
    //[infer]~^ ERROR expected generic lifetime parameter, found `'_`

    #[cfg(fixed)]
    let i: i32 = foo::<'a>();

    i
}

fn main() {}
