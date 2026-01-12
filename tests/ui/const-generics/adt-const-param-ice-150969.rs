//@ ignore-backends: gcc
#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete
#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete

fn pass_enum<const N : usize, const M : usize = const {N}>() {
    //~^ ERROR defaults for generic parameters are not allowed here
    pass_enum::<{None}>();
    //~^ ERROR missing generics for enum `Option`
}

fn main() {}
