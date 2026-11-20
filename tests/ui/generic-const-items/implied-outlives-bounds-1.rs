// Check that paths referring to (generic) free const items induce implied bounds.
//@ revisions: pass fail dump
//@[pass] check-pass
#![feature(generic_const_items, gca_min_const_items)]
#![cfg_attr(dump, feature(rustc_attrs))]

// References of `EXP::<'r, 'q>` (`'r`, `'q` early-bound) should induce implied bound `'q: 'r` in
// the enclosing definition due to the explicit outlives-bound `'b: 'a` on `EXP`.
const EXP<'a, 'b: 'a>: usize = std::gca!(0);

// We imply `'b: 'a` from const alias `EXP::<'a, 'b>` in the constant's body.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
const CT0<'a, 'b>: usize = std::gca!(EXP::<'a, 'b>);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: 'b: 'a

// We imply `'b: 'a` from const alias `EXP::<'a, 'b>` in the struct's signature.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
struct Ty0<'a, 'b>([(); std::gca!(EXP::<'a, 'b>)]);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: 'b: 'a

// References of `IMP::<'r, 'q>` (`'r`, `'q` early-bound) should induce implied bound `'q: 'r` in
// the enclosing definition due to the implicit outlives-bound `'b: 'a` on `IMP`.
const IMP<'a, 'b>: usize = std::gca!(EXP::<'a, 'b>);

// We imply `'b: 'a` from const alias `EXP::<'a, 'b>` in the constant's body.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
const CT1<'a, 'b>: usize = std::gca!(IMP::<'a, 'b>);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: 'b: 'a

// We imply `'b: 'a` from const alias `EXP::<'a, 'b>` in the struct's signature.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
struct Ty1<'a, 'b>([(); std::gca!(IMP::<'a, 'b>)]);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: 'b: 'a

const CT2<'a, 'b>: [(); std::gca!(EXP::<'a, 'b>)] = {
    let _: &'a &'b (); // OK since `EXP::<'a, 'b>` implies `'b: 'a`
    []
};

// Ensure that we actually enforce these implied bounds at usage sites:

#[cfg(fail)] fn env0<'any>() { let _ = CT0::<'static, 'any>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env1<'any>() { let _: Ty0<'static, 'any>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env2<'any>() { let _ = CT1::<'static, 'any>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env3<'any>() { let _ = Ty1::<'static, 'any>; }
//[fail]~^ ERROR lifetime may not live long enough


fn main() {}
