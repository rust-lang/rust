// FIXME(fmease): This still needs to be implemented
//@ ignore-test
// Check that paths referring to (generic) free const items induce implied bounds.
//@ revisions: pos neg
//@[pos] check-pass
#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_items)]

// References of `EXP::<'r, 'q>` should induce implied bound `'q: 'r` in the enclosing def.
type const EXP<'a, 'b: 'a>: usize = 0;

struct Ty<'a, 'b>([(); EXP::<'a, 'b>]); // we imply `'a: 'b`
const CT<'a, 'b>: [(); EXP::<'a, 'b>] = {
    let _: &'a &'b (); // OK since `EXP::<'a, 'b>` implies `'a: 'b`
    []
};

#[cfg(neg)]
fn env0<'any>() {
    let _: Ty<'static, 'any>; //[neg]~ ERROR lifetime may not live long enough
}

#[cfg(neg)]
fn env1<'any>() {
    _ = CT::<'static, 'any>; //[neg]~ ERROR lifetime may not live long enough
}
