//! Check that we suggest `cfg(any())` -> `false` and `cfg(all())` -> true
//! Additionally tests the behaviour of empty cfg predicates.
#![deny(empty_cfg_predicate)]

#[cfg(any())]  //~ ERROR: use of empty `cfg(any())`
struct A;
#[cfg(all())]  //~ ERROR: use of empty `cfg(all())`
struct B;

fn main() {
    // cfg-d out by `any()`
    A; //~ ERROR: cannot find value `A`
    // OK: `all()` evaluates to true
    B;
}
