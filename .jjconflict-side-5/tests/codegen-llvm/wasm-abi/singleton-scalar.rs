//@ add-minicore
//@ revisions: wasm wasip1
//@[wasm] compile-flags: --target wasm32-unknown-unknown
//@[wasip1] compile-flags: --target wasm32-wasip1
//@ needs-llvm-components: webassembly
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled -Ctarget-feature=+simd128
#![feature(no_core, rustc_attrs, f128)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct ReprC<T>(T);

#[repr(C, align(8))]
struct ReprCAlign8<T>(T);

#[repr(transparent)]
struct ReprTransparent<T>(T);

#[repr(C)]
struct TwoMemberStruct<T> {
    a: T,
    b: T,
}

#[repr(C)]
union TwoMemberUnion<T: Copy> {
    a: T,
    b: T,
}

#[repr(C)]
struct CheckSibling<T, U> {
    x: T,
    y: U,
}

#[repr(C)]
union CheckSiblingUnion<T: Copy, U: Copy> {
    x: T,
    y: U,
}

#[repr(C, align(16))]
struct AlignedUnit;

mod pass_i32 {
    use super::*;

    // CHECK: define{{.*}} i32 @pass_i32(i32 noundef returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32(x: i32) -> i32 {
        x
    }

    // CHECK: define{{.*}} i32 @pass_transparent_i32(i32 noundef returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_i32(x: ReprTransparent<i32>) -> ReprTransparent<i32> {
        x
    }

    // CHECK: define{{.*}} i32 @pass_array_i32(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_array_i32(x: [i32; 1]) -> [i32; 1] {
        x
    }

    // CHECK: define{{.*}} i32 @pass_c_i32(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_c_i32(x: ReprC<i32>) -> ReprC<i32> {
        x
    }

    // CHECK: define{{.*}} i32 @pass_maybe_uninit_i32(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_maybe_uninit_i32(x: MaybeUninit<i32>) -> MaybeUninit<i32> {
        x
    }

    // CHECK: define{{.*}} void @pass_two_member_union_i32(ptr{{.*}} sret([4 x i8]){{.*}}, ptr{{.*}} dereferenceable(4) %x)
    #[unsafe(no_mangle)]
    extern "C" fn pass_two_member_union_i32(x: TwoMemberUnion<i32>) -> TwoMemberUnion<i32> {
        x
    }

    // CHECK: define{{.*}} void @pass_two_member_struct_i32(ptr{{.*}} sret([8 x i8]){{.*}}, ptr{{.*}} dereferenceable(8) %x)
    #[unsafe(no_mangle)]
    extern "C" fn pass_two_member_struct_i32(x: TwoMemberStruct<i32>) -> TwoMemberStruct<i32> {
        x
    }

    // A zero-sized struct is ignored, the struct is still a trivial aggregate.
    //
    // CHECK: define{{.*}} i32 @pass_i32_and_zst_struct(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_and_zst_struct(x: CheckSibling<i32, ()>) -> CheckSibling<i32, ()> {
        x
    }

    // Unless it over-aligns.
    //
    // CHECK: define{{.*}} void @pass_i32_and_zst_struct_overaligned(ptr{{.*}}, ptr{{.*}})
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_and_zst_struct_overaligned(
        x: CheckSibling<i32, AlignedUnit>,
    ) -> CheckSibling<i32, AlignedUnit> {
        x
    }

    // A zero-sized array is ignored, the struct is still a trivial aggregate.
    //
    // CHECK: define{{.*}} i32 @pass_i32_and_zst_array(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_and_zst_array(
        x: CheckSibling<i32, [u32; 0]>,
    ) -> CheckSibling<i32, [u32; 0]> {
        x
    }

    // CHECK: define{{.*}} i32 @pass_i32_and_zst_array_underaligned(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_and_zst_array_underaligned(
        x: CheckSibling<i32, [u8; 0]>,
    ) -> CheckSibling<i32, [u8; 0]> {
        x
    }

    // Unless it over-aligns.
    //
    // CHECK: define{{.*}} void @pass_i32_and_zst_array_overaligned(ptr{{.*}}, ptr{{.*}})
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_and_zst_array_overaligned(
        x: CheckSibling<i32, [u64; 0]>,
    ) -> CheckSibling<i32, [u64; 0]> {
        x
    }

    // CHECK: define{{.*}} i32 @pass_i32_or_zst_struct(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_or_zst_struct(
        x: CheckSiblingUnion<i32, ()>,
    ) -> CheckSiblingUnion<i32, ()> {
        x
    }

    // CHECK: define{{.*}} i32 @pass_i32_or_zst_array(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_or_zst_array(
        x: CheckSiblingUnion<i32, [i32; 0]>,
    ) -> CheckSiblingUnion<i32, [i32; 0]> {
        x
    }

    // CHECK: define{{.*}} void @pass_c_i32_overaligned(ptr{{.*}} sret([8 x i8]){{.*}}, ptr{{.*}} dereferenceable(8) %x)
    #[unsafe(no_mangle)]
    extern "C" fn pass_c_i32_overaligned(x: ReprCAlign8<i32>) -> ReprCAlign8<i32> {
        x
    }

    // CHECK: define{{.*}} i32 @pass_i32_recursive(i32 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_recursive(
        x: ReprTransparent<ReprC<[i32; 1]>>,
    ) -> ReprTransparent<ReprC<[i32; 1]>> {
        x
    }

    #[repr(i32)]
    enum CLikeIntEnum {
        A,
        B,
    }

    // CHECK: define{{.*}} i32 @pass_i32_c_like_enum(i32 noundef returned range(i32 0, 2) %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_c_like_enum(x: CLikeIntEnum) -> CLikeIntEnum {
        x
    }

    // CHECK: define{{.*}} i32 @pass_transparent_i32_c_like_enum(i32 noundef returned range(i32 0, 2) %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_i32_c_like_enum(
        x: ReprTransparent<CLikeIntEnum>,
    ) -> ReprTransparent<CLikeIntEnum> {
        x
    }

    #[repr(i32)]
    enum IntEnumZstStructVariants {
        A(()),
        B(),
    }

    // Any field, even a ZST, disqualifies an enum from being passed as a scalar.
    //
    // CHECK: define{{.*}} void @pass_i32_enum_zst_struct_variants(ptr{{.*}}, ptr{{.*}})
    #[unsafe(no_mangle)]
    extern "C" fn pass_i32_enum_zst_struct_variants(
        x: IntEnumZstStructVariants,
    ) -> IntEnumZstStructVariants {
        x
    }

    // CHECK: define{{.*}} void @pass_transparent_i32_enum_zst_struct_variants(ptr{{.*}}, ptr{{.*}})
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_i32_enum_zst_struct_variants(
        x: ReprTransparent<IntEnumZstStructVariants>,
    ) -> ReprTransparent<IntEnumZstStructVariants> {
        x
    }

    // CHECK: define{{.*}} void @pass_c_i32_enum_zst_struct_variants(ptr{{.*}}, ptr{{.*}})
    #[unsafe(no_mangle)]
    extern "C" fn pass_c_i32_enum_zst_struct_variants(
        x: ReprC<IntEnumZstStructVariants>,
    ) -> ReprC<IntEnumZstStructVariants> {
        x
    }
}

mod pass_ptr {
    use super::*;

    // The layout of `Option<&T>` is guaranteed to match `*const T`.
    //
    // CHECK: define{{.*}} ptr @pass_option_ref(ptr{{.*}} %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_option_ref(x: Option<&'static i32>) -> Option<&'static i32> {
        x
    }

    // CHECK: define{{.*}} ptr @pass_transparent_option_ref(ptr{{.*}} %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_option_ref(
        x: ReprTransparent<Option<&'static i32>>,
    ) -> ReprTransparent<Option<&'static i32>> {
        x
    }
}

mod pass_simd {
    use super::*;

    // CHECK: define{{.*}} <4 x float> @pass_simd_f32x4(<4 x float> returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_simd_f32x4(x: simd::f32x4) -> simd::f32x4 {
        x
    }

    // CHECK: define{{.*}} <4 x float> @pass_transparent_simd_f32x4(<4 x float> returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_simd_f32x4(
        x: ReprTransparent<simd::f32x4>,
    ) -> ReprTransparent<simd::f32x4> {
        x
    }
}

mod pass_i128 {
    use super::*;

    // CHECK: define{{.*}} void @pass_i128(ptr{{.*}}, i128 noundef %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_i128(x: i128) -> i128 {
        x
    }

    // CHECK: define{{.*}} void @pass_transparent_i128(ptr{{.*}}, i128 noundef %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_i128(x: ReprTransparent<i128>) -> ReprTransparent<i128> {
        x
    }

    // CHECK: define{{.*}} i128 @pass_c_i128(i128 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_c_i128(x: ReprC<i128>) -> ReprC<i128> {
        x
    }

    // CHECK: define{{.*}} void @pass_maybe_uninit_i128(ptr{{.*}}, i128 %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_maybe_uninit_i128(x: MaybeUninit<i128>) -> MaybeUninit<i128> {
        x
    }

    // CHECK: define{{.*}} void @pass_two_member_union_i128(ptr{{.*}} sret([16 x i8]){{.*}}, ptr{{.*}} dereferenceable(16) %x)
    #[unsafe(no_mangle)]
    extern "C" fn pass_two_member_union_i128(x: TwoMemberUnion<i128>) -> TwoMemberUnion<i128> {
        x
    }
}

mod pass_f32 {
    use super::*;

    // CHECK: define{{.*}} float @pass_f32(float noundef returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_f32(x: f32) -> f32 {
        x
    }

    // CHECK: define{{.*}} float @pass_transparent_f32(float noundef returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_f32(x: ReprTransparent<f32>) -> ReprTransparent<f32> {
        x
    }

    // CHECK: define{{.*}} float @pass_c_f32(float returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_c_f32(x: ReprC<f32>) -> ReprC<f32> {
        x
    }

    // CHECK: define{{.*}} float @pass_maybe_uninit_f32(float returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_maybe_uninit_f32(x: MaybeUninit<f32>) -> MaybeUninit<f32> {
        x
    }

    // CHECK: define{{.*}} void @pass_two_member_union_f32(ptr{{.*}} sret([4 x i8]){{.*}}, ptr{{.*}} dereferenceable(4) %x)
    #[unsafe(no_mangle)]
    extern "C" fn pass_two_member_union_f32(x: TwoMemberUnion<f32>) -> TwoMemberUnion<f32> {
        x
    }
}

mod pass_f128 {
    use super::*;

    // CHECK: define{{.*}} void @pass_f128(ptr{{.*}}, fp128 noundef %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_f128(x: f128) -> f128 {
        x
    }

    // CHECK: define{{.*}} void @pass_transparent_f128(ptr{{.*}}, fp128 noundef %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_transparent_f128(x: ReprTransparent<f128>) -> ReprTransparent<f128> {
        x
    }

    // CHECK: define{{.*}} fp128 @pass_c_f128(fp128 returned %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_c_f128(x: ReprC<f128>) -> ReprC<f128> {
        x
    }

    // CHECK: define{{.*}} void @pass_maybe_uninit_f128(ptr{{.*}}, fp128 %[[ARG:.*]])
    #[unsafe(no_mangle)]
    extern "C" fn pass_maybe_uninit_f128(x: MaybeUninit<f128>) -> MaybeUninit<f128> {
        x
    }

    // CHECK: define{{.*}} void @pass_two_member_union_f128(ptr{{.*}} sret([16 x i8]){{.*}}, ptr{{.*}} dereferenceable(16) %x)
    #[unsafe(no_mangle)]
    extern "C" fn pass_two_member_union_f128(x: TwoMemberUnion<f128>) -> TwoMemberUnion<f128> {
        x
    }
}
