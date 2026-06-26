//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]

#[no_mangle]
pub fn demo_for_i32() {
    generic_impl::<i32>();
}

// Two important things here:
// - We replace the "then" block with `unreachable` to avoid linking problems
// - We neither declare nor define the `big_impl` that said block "calls".

// CHECK-LABEL: ; skip_mono_inside_if_false::generic_impl
// CHECK: start:
// CHECK-NEXT: br label %[[ELSE_BRANCH:bb[0-9]+]]
// CHECK: [[ELSE_BRANCH]]:
// CHECK-NEXT: call skip_mono_inside_if_false::small_impl
// CHECK: bb{{[0-9]+}}:
// CHECK-NEXT: ret void
// CHECK: bb{{[0-9+]}}:
// CHECK-NEXT: unreachable

fn generic_impl<T>() {
    trait MagicTrait {
        const IS_BIG: bool;
    }
    impl<T> MagicTrait for T {
        const IS_BIG: bool = std::mem::size_of::<T>() > 10;
    }
    if T::IS_BIG {
        big_impl::<T>();
    } else {
        small_impl::<T>();
    }
}

#[inline(never)]
fn small_impl<T>() {}
#[inline(never)]
fn big_impl<T>() {}
