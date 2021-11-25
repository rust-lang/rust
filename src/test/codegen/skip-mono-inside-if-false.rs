// compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

#[no_mangle]
pub fn demo_for_i32() {
    generic_impl::<i32>();
}

// CHECK-LABEL: ; skip_mono_inside_if_false::generic_impl
// CHECK: start:
// CHECK-NEXT: br i1 false, label %[[THEN_BRANCH:bb[0-9]+]], label %[[ELSE_BRANCH:bb[0-9]+]]
// CHECK: [[ELSE_BRANCH]]:
// CHECK-NEXT: call skip_mono_inside_if_false::small_impl
// CHECK: [[THEN_BRANCH]]:
// CHECK-NEXT: call skip_mono_inside_if_false::big_impl

// Then despite there being calls to both of them, only the ones that's used has a definition.
// The other is only forward-declared, and its use will disappear with LLVM's simplifycfg.

// CHECK: define internal void @_ZN25skip_mono_inside_if_false10small_impl
// CHECK: declare hidden void @_ZN25skip_mono_inside_if_false8big_impl

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
