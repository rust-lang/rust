//@ revisions: debug-assertions-add no-debug-assertions-add debug-assertions-and no-debug-assertions-and debug-assertions-not no-debug-assertions-not
// Issue: #165685
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

const GCA_U64<const N: u64>: usize = gca!(INNER::<N>);
//~^ ERROR the constant `N` is not of type `usize`
//~| ERROR the constant `INNER::<N>` is not of type `usize`

const GCA_CONCRETE_USIZE<const N: ConcreteUsize>: usize = gca!(INNER::<N>);
//~^ ERROR the constant `N` is not of type `usize`
//~| ERROR the constant `INNER::<N>` is not of type `usize`

const ARR_U64: [(); gca!(INNER::<0>)] = [(); gca!(GCA_U64::<0>)];
//~^ ERROR type mismatch resolving `GCA_U64<0> == _` [E0271]
const ARR_CONCRETE_USIZE: [(); gca!(INNER::<0>)] = [(); gca!(GCA_CONCRETE_USIZE::<0>)];
//~^ ERROR type mismatch resolving `GCA_CONCRETE_USIZE<0> == _` [E0271]

fn main() {}
