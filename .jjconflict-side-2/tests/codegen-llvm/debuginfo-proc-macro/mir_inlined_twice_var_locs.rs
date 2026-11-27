//@ compile-flags: -Cdebuginfo=2 -Copt-level=0 -Zmir-enable-passes=+Inline
// MSVC is different because of the individual allocas.
//@ ignore-msvc

//@ proc-macro: macro_def.rs

// Find the variable.
// CHECK-DAG: ![[#var_dbg:]] = !DILocalVariable(name: "n",{{( arg: 1,)?}} scope: ![[#var_scope:]]

// Find both dbg_declares. These will proceed the variable metadata, of course, so we're looking
// backwards.
// CHECK-DAG: dbg_declare(ptr %n.dbg.spill{{[0-9]}}, ![[#var_dbg]], !DIExpression(), ![[#var_loc2:]])
// CHECK-DAG: dbg_declare(ptr %n.dbg.spill, ![[#var_dbg]], !DIExpression(), ![[#var_loc1:]])

// Find the first location definition, looking forwards again.
// CHECK: ![[#var_loc1]] = !DILocation
// CHECK-SAME: scope: ![[#var_scope:]], inlinedAt: ![[#var_inlinedAt1:]]

// Find the first location's inlinedAt
// NB: If we fail here it's *probably* because we failed to produce two
// different locations and ended up reusing an earlier one.
// CHECK: ![[#var_inlinedAt1]] = !DILocation
// CHECK-SAME: scope: ![[var_inlinedAt1_scope:]]

// Find the second location definition, still looking forwards.
// NB: If we failed to produce two different locations, the test will
// definitely fail by this point (if it hasn't already) because we won't
// be able to find the same line again.
// CHECK: ![[#var_loc2]] = !DILocation
// CHECK-SAME: scope: ![[#var_scope]], inlinedAt: ![[#var_inlinedAt2:]]

// Find the second location's inlinedAt.
// CHECK: ![[#var_inlinedAt2]] = !DILocation
// CHECK-SAME: scope: ![[#var_inlinedAt2_scope:]]

// Finally, check that a discriminator was emitted for the second scope.
// FIXMEkhuey ideally we would check that *either* scope has a discriminator
// but I don't know that it's possible to check that with FileCheck.
// CHECK: ![[#var_inlinedAt2_scope]] = !DILexicalBlockFile
// CHECK-SAME: discriminator: [[#]]
extern crate macro_def;

use std::env;

fn square(n: i32) -> i32 {
    n * n
}

fn main() {
    let (z1, z2) = macro_def::square_twice!();
    println!("{z1} == {z2}");
}
