// compile-flags: -Zsave-analysis

#![feature(type_alias_impl_trait)]

type Closure = impl FnOnce(); //~ ERROR: type mismatch resolving

fn c() -> Closure {
    || -> Closure { || () }
}

fn main() {}
