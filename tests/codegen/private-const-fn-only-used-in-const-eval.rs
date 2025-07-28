//! Check that we — where possible — don't codegen functions that are only used to evaluate
//! static/const items, but never used in runtime code.

//@ compile-flags: --crate-type=lib -Copt-level=0

#![feature(generic_const_items)] // only used in the last few test cases

pub static STATIC: () = func0();
const fn func0() {}
// CHECK-NOT: define{{.*}}func0{{.*}}

pub const CONSTANT: () = func1();
const fn func1() {}
// CHECK-NOT: define{{.*}}func1{{.*}}

// The const item is impossible to reference and thus, we don't need to codegen `func2`.
pub const IMPOSS_TO_REF: fn() = func2
where
    for<'_delay> String: Copy;
const fn func2() {}
// CHECK-NOT: define{{.*}}func2{{.*}}

// We generally don't want to evaluate the initializer of free const items if the latter have
// non-region params (and even if we did, const eval would fail anyway with "too polymorphic"
// if the initializer actually referenced such a param).
//
// As a result of not being able to look at the final value, during reachability analysis we
// can't tell for sure if for example certain functions end up in the final value or if they're
// only used during const eval. We fall back to a conservative HIR-based approach.

// `func3` isn't needed at runtime but the compiler can't tell for the reason mentioned above.
pub const POLY_CONST_0<const C: bool>: () = func3();
const fn func3() {}
// CHECK: define{{.*}}func3{{.*}}

// `func4` isn't needed at runtime but the compiler can't tell for the reason mentioned above.
pub const POLY_CONST_1<const C: bool>: () = if C { func4() };
const fn func4() {}
// CHECK: define{{.*}}func4{{.*}}

// `func5` *is* needed at runtime (here, the HIR-based approach gets it right).
pub const POLY_CONST_2<const C: bool>: Option<fn() /* or a TAIT */> =
    if C { Some(func5) } else { None };
const fn func5() {}
// CHECK: define{{.*}}func5{{.*}}
