//! Check that we — where possible — don't codegen functions that are only used to evaluate
//! static/const items, but never used in runtime code.

//@ compile-flags: --crate-type=lib -Copt-level=0

#![feature(generic_const_items)] // only used for latter test cases

const fn foo() {}
pub static FOO: () = foo();
// CHECK-NOT: define{{.*}}foo{{.*}}

const fn bar() {}
pub const BAR: () = bar();
// CHECK-NOT: define{{.*}}bar{{.*}}

// We don't / can't evaluate the const initializer of `BAZ` because it's too generic. As a result,
// we can't tell during reachability analysis if `baz` ends up in the final value or if it's only
// used during const eval (just by looking at it here, it's obvious that it isn't needed at runtime;
// it's a limitation of the compiler that it can't figure that out).
const fn baz() {}
pub const BAZ<const C: bool>: () = if C { baz() };
// CHECK: define{{.*}}baz{{.*}}

// The const `BAN` is impossible to reference and thus, we don't need to codegen `ban`.
const fn ban() {}
pub const BAN: fn() = ban
where
    for<'_delay> String: Copy;
// CHECK-NOT: define{{.*}}ban{{.*}}
