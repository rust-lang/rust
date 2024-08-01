//@ known-bug: #119924
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl, effects)]

struct S;
#[const_trait]
trait Trait<const N: u32> {}

const fn f<T: Trait<{
    struct I<U: ~const Trait<0>>(U); // should've gotten rejected during AST validation
    //~^ ICE no host param id for call in const yet no errors reported
    0
}>>() {}

pub fn main() {}
