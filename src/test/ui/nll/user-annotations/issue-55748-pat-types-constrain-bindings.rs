// This test is ensuring that type ascriptions on let bindings
// constrain both:
//
// 1. the input expression on the right-hand side (after any potential
//    coercion, and allowing for covariance), *and*
//
// 2. the bindings (if any) nested within the pattern on the left-hand
//    side (and here, the type-constraint is *invariant*).

#![feature(nll)]

#![allow(dead_code, unused_mut)]
type PairUncoupled<'a, 'b, T> = (&'a T, &'b T);
type PairCoupledRegions<'a, T> = (&'a T, &'a T);
type PairCoupledTypes<T> = (T, T);

fn uncoupled_lhs<'a>(_x: &'a u32, s: &'static u32) -> &'static u32 {
    let ((mut y, mut _z),): (PairUncoupled<u32>,) = ((s, &_x),); // ok
    // Above compiling does *not* imply below would compile.
    // ::std::mem::swap(&mut y, &mut _z);
    y
}

fn swap_regions((mut y, mut _z): PairCoupledRegions<u32>) {
    ::std::mem::swap(&mut y, &mut _z);
}

fn coupled_regions_lhs<'a>(_x: &'a u32, s: &'static u32) -> &'static u32 {
    let ((y, _z),): (PairCoupledRegions<u32>,) = ((s, &_x),);
    // If above line compiled, so should line below ...

    // swap_regions((y, _z));

    // ... but the ascribed type also invalidates this use of `y`
    y //~ ERROR lifetime may not live long enough
}

fn swap_types((mut y, mut _z): PairCoupledTypes<&u32>) {
    ::std::mem::swap(&mut y, &mut _z);
}

fn coupled_types_lhs<'a>(_x: &'a u32, s: &'static u32) -> &'static u32 {
    let ((y, _z),): (PairCoupledTypes<&u32>,) = ((s, &_x),);
    // If above line compiled, so should line below ...

    // swap_types((y, _z));

    // ... but the ascribed type also invalidates this use of `y`
    y //~ ERROR lifetime may not live long enough
}

fn swap_wilds((mut y, mut _z): PairCoupledTypes<&u32>) {
    ::std::mem::swap(&mut y, &mut _z);
}

fn coupled_wilds_lhs<'a>(_x: &'a u32, s: &'static u32) -> &'static u32 {
    let ((y, _z),): (PairCoupledTypes<_>,) = ((s, &_x),);
    // If above line compiled, so should line below
    // swap_wilds((y, _z));

    // ... but the ascribed type also invalidates this use of `y`
    y //~ ERROR lifetime may not live long enough
}

fn main() {
    uncoupled_lhs(&3, &4);
    coupled_regions_lhs(&3, &4);
    coupled_types_lhs(&3, &4);
    coupled_wilds_lhs(&3, &4);
}
