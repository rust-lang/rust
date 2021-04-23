// compile-flags: -Zsave-analysis

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type Closure = impl FnOnce(); //~ ERROR: type mismatch resolving

fn c() -> Closure {
    || -> Closure { || () } //[min_tait]~ ERROR: not permitted here
}

fn main() {}
