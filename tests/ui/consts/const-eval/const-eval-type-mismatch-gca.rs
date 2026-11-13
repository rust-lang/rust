//@ revisions: debug-assertions-add no-debug-assertions-add debug-assertions-and no-debug-assertions-and debug-assertions-not no-debug-assertions-not
// This test verifies ICE doesn't happen on mismatched types of the result of binary
// or unary operations during const eval. See Issue #165685.
//@ [no-debug-assertions-add] compile-flags: -Znext-solver=globally -Cdebug_assertions=no
//@ [no-debug-assertions-and] compile-flags: -Znext-solver=globally -Cdebug_assertions=no
//@ [no-debug-assertions-not] compile-flags: -Znext-solver=globally -Cdebug_assertions=no
//@ [debug-assertions-add]    compile-flags: -Znext-solver=globally -Cdebug_assertions=yes
//@ [debug-assertions-and]    compile-flags: -Znext-solver=globally -Cdebug_assertions=yes
//@ [debug-assertions-not]    compile-flags: -Znext-solver=globally -Cdebug_assertions=yes

#![feature(generic_const_items)]
#![feature(gca_min_const_items)]
#![feature(gca_const_items)]
#![expect(incomplete_features)]

use std::gca;

type ConcreteUsize = cfg_select! {
    target_pointer_width = "16" => u16,
    target_pointer_width = "32" => u32,
    target_pointer_width = "64" => u64,
};

const INNER<const N: usize>: usize = cfg_select! {
    // Test the binary operator `dest = left OP right` where `dest.ty = left.ty` should hold.
    any(debug_assertions_add, no_debug_assertions_add) => N + 1,
    // Test the binary operator `dest = left OP right` where `right.ty = left.ty` should hold.
    any(debug_assertions_and, no_debug_assertions_and) => N & 1,
    // Test the binary operator `dest = OP operand` where `dest.ty = operand.ty` should hold.
    any(debug_assertions_not, no_debug_assertions_not) => !N,
};

const GCA<const N: ConcreteUsize>: usize = gca!(INNER::<N>);
//~^ ERROR the constant `N` is not of type `usize`
//~| ERROR the constant `INNER::<N>` is not of type `usize`

const ARR: [(); gca!(INNER::<0>)] = [(); gca!(GCA::<0>)];
//~^ ERROR type mismatch resolving `GCA<0> == _` [E0271]

fn main() {}
