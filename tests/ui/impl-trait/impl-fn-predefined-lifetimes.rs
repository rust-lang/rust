//@ revisions: edition2015 edition2024 next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[edition2015] edition:2015
//@[edition2024] edition:2024
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![feature(impl_trait_in_fn_trait_return)]

// In the old solver we only define the nested `impl Sized`
// inside of the closure, at which point `'a` is an external
// region. The edition does not matter except for the diagnostic.
// In both cases the inner opaque type captures `'a`.

fn a<'a>() -> impl Fn(&'a u8) -> (impl Sized + 'a) {
    |x| x
    //[edition2015,edition2024]~^ ERROR expected generic lifetime parameter, found `'_`
}

fn _b<'a>() -> impl Fn(&'a u8) -> (impl Sized + 'a) {
    a()
}

// `'_` gets inferred to `'a` here.
fn c<'a>() -> impl Fn(&'a u8) -> (impl Sized + '_) {
    |x| x
    //[edition2015,edition2024]~^ ERROR expected generic lifetime parameter, found `'_`
}

fn _d<'a>() -> impl Fn(&'a u8) -> (impl Sized + 'a) {
    a()
}

fn main() {}
