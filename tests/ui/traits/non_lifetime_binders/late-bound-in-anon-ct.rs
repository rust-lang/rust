#![feature(non_lifetime_binders, generic_const_exprs)]
//~^ WARN the feature `non_lifetime_binders` is incomplete
//~| WARN the feature `generic_const_exprs` is incomplete

fn foo() -> usize
where
    for<T> [i32; { let _: T = todo!(); 0 }]:,
    //~^ ERROR cannot capture late-bound type parameter in a constant
{}

fn main() {}
