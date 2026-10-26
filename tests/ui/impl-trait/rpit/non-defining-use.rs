//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
// Regression test for #111935 that non-defining uses of RPIT result in errors
#![allow(unconditional_recursion)]
fn foo<T>() -> impl Sized {
    let _: () = foo::<u8>(); //[current]~ ERROR expected generic type parameter, found `u8`
}

fn bar<T>(val: T) -> impl Sized {
    let _: u8 = bar(0u8);
    //[current]~^ ERROR expected generic type parameter, found `u8`
    val
}

fn main() {}
