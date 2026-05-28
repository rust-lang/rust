// ICE no entry found for key generics_of
// issue: rust-lang/rust#113133

#![allow(incomplete_features)]
#![feature(generic_const_exprs, non_lifetime_binders)]

pub fn foo()
where
    for<const N: usize = { const fn bar() {} bar(); 1 }> ():,
    //~^ ERROR late-bound const parameters cannot be used currently
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
{}

fn main() {}
