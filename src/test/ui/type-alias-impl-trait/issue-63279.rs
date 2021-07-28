// compile-flags: -Zsave-analysis

#![feature(type_alias_impl_trait)]

type Closure = impl FnOnce();

fn c() -> Closure {
    || -> Closure { || () } //~ ERROR: mismatched types
}

fn main() {}
