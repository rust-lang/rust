// compile-flags: -C no-prepopulate-passes
// ignore-tidy-linelength
// min-llvm-version 7.0

#![crate_type = "lib"]

// Below, these constants are defined as enum variants that by itself would
// have a lower alignment than the enum type. Ensure that we mark them
// correctly with the higher alignment of the enum.

// CHECK: @STATIC = {{.*}}, align 4

// This checks the constants from inline_enum_const
// CHECK: @{{[0-9]+}} = {{.*}}, align 2

// This checks the constants from {low,high}_align_const, they share the same
// constant, but the alignment differs, so the higher one should be used
// CHECK: [[LOW_HIGH:@[0-9]+]] = {{.*}}, align 4

#[derive(Copy, Clone)]

// repr(i16) is required for the {low,high}_align_const test
#[repr(i16)]
pub enum E<A, B> {
    A(A),
    B(B),
}

#[no_mangle]
pub static STATIC: E<i16, i32> = E::A(0);

// CHECK-LABEL: @static_enum_const
#[no_mangle]
pub fn static_enum_const() -> E<i16, i32> {
   STATIC
}

// CHECK-LABEL: @inline_enum_const
#[no_mangle]
pub fn inline_enum_const() -> E<i8, i16> {
    *&E::A(0)
}

// CHECK-LABEL: @low_align_const
#[no_mangle]
pub fn low_align_const() -> E<i16, [i16; 3]> {
// Check that low_align_const and high_align_const use the same constant
// CHECK: i8* align 2 getelementptr inbounds (<{ [8 x i8] }>, <{ [8 x i8] }>* [[LOW_HIGH]], i32 0, i32 0, i32 0),
    *&E::A(0)
}

// CHECK-LABEL: @high_align_const
#[no_mangle]
pub fn high_align_const() -> E<i16, i32> {
// Check that low_align_const and high_align_const use the same constant
// CHECK: i8* align 4 getelementptr inbounds (<{ [8 x i8] }>, <{ [8 x i8] }>* [[LOW_HIGH]], i32 0, i32 0, i32 0),
    *&E::A(0)
}
