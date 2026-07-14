//@ check-pass
//@ compile-flags: -Znext-solver=globally
//@ dont-require-annotations: NOTE

#![feature(min_generic_const_args, generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    const ASSOC: usize;
}

fn foo(_: &dyn Trait<ASSOC = 10>) {}

fn main() {}
