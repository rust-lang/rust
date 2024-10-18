//@ compile-flags: -Cno-prepopulate-passes -O
//@ only-64bit (I don't care about alignment under different bits)

// Testing different niches updates to the corresponding alignment.

#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(never_type)]

#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(0x7fff)]
struct RestrictedAddress_0_0x7fff(&'static i64);

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(0x7fff)]
struct RestrictedAddress_1_0x7fff(&'static i64);

#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(0xf000)]
struct RestrictedAddress_0_0xf000(&'static i64);

enum MultipleAlign8 {
    Untag(RestrictedAddress_1_0x7fff),
    Niche_32768,
    Uninhabited_1(!),
    Uninhabited_2(!),
    Uninhabited_3(!),
    Uninhabited_4(!),
    Uninhabited_5(!),
    Uninhabited_6(!),
    Uninhabited_7(!),
    Niche_32776,
}

// CHECK-LABEL: @multiple_niches_align_8
// CHECK-SAME: align 8 {{.*}}%a)
#[no_mangle]
#[inline(never)]
fn multiple_niches_align_8(a: MultipleAlign8) {}

// CHECK-LABEL: @call_multiple_niches_align_8
#[no_mangle]
fn call_multiple_niches_align_8() {
    // CHECK: call void @multiple_niches_align_8(ptr {{.*}}align 8 {{.*}}(i64 32768 to ptr)
    multiple_niches_align_8(MultipleAlign8::Niche_32768);
    // CHECK: call void @multiple_niches_align_8(ptr {{.*}}align 8 {{.*}}(i64 32776 to ptr)
    multiple_niches_align_8(MultipleAlign8::Niche_32776);
}

enum MultipleAlign2 {
    Untag(RestrictedAddress_1_0x7fff),
    Niche_32768,
    Uninhabited_1(!),
    Niche_32770,
}

// CHECK-LABEL: @multiple_niches_align_2
// CHECK-SAME: align 2 {{.*}}%a)
#[no_mangle]
#[inline(never)]
fn multiple_niches_align_2(a: MultipleAlign2) {}

// CHECK-LABEL: @call_multiple_niches_align_2
#[no_mangle]
fn call_multiple_niches_align_2() {
    // CHECK: call void @multiple_niches_align_2(ptr {{.*}}align 2 {{.*}}(i64 32768 to ptr)
    multiple_niches_align_2(MultipleAlign2::Niche_32768);
    // CHECK: call void @multiple_niches_align_2(ptr {{.*}}align 2 {{.*}}(i64 32770 to ptr)
    multiple_niches_align_2(MultipleAlign2::Niche_32770);
}

enum SingleAlign8 {
    Untag(RestrictedAddress_0_0x7fff),
    Niche_32768,
}

// CHECK-LABEL: @single_niche_align_8
// CHECK-SAME: align 8 {{.*}}%a)
#[no_mangle]
#[inline(never)]
fn single_niche_align_8(a: SingleAlign8) {}

// CHECK-LABEL: @call_single_niche_align_8
#[no_mangle]
fn call_single_niche_align_8() {
    // CHECK: call void @single_niche_align_8(ptr {{.*}}align 8 {{.*}}(i64 32768 to ptr)
    single_niche_align_8(SingleAlign8::Niche_32768);
}

enum SingleAlign1 {
    Untag(RestrictedAddress_0_0xf000),
    Niche_61441,
}

// CHECK-LABEL: @single_niche_align_1
// CHECK-SAME: align 1 {{.*}}%a)
#[no_mangle]
#[inline(never)]
fn single_niche_align_1(a: SingleAlign1) {}

// CHECK-LABEL: @call_single_niche_align_1
#[no_mangle]
fn call_single_niche_align_1() {
    // CHECK: call void @single_niche_align_1(ptr {{.*}}align 1 {{.*}}(i64 61441 to ptr)
    single_niche_align_1(SingleAlign1::Niche_61441);
}

// Check that we only apply the new alignment on enum.

// CHECK-LABEL: @restricted_address
// CHECK-SAME: align 8 {{.*}}%a)
#[no_mangle]
fn restricted_address(a: RestrictedAddress_0_0x7fff) {}
