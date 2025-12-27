//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

#![crate_type = "lib"]

#[derive(PartialEq)]
pub struct NoPadding {
    x: u16,
    y: u16,
}

// CHECK-LABEL: fn cmp_no_padding(
fn cmp_no_padding(a: &NoPadding, b: &NoPadding) -> bool {
    // CHECK: compare_bitwise::<NoPadding>(
    a == b
}
// EMIT_MIR derived_partialeq.cmp_no_padding.PreCodegen.after.mir

#[derive(PartialEq)]
pub struct Recursive {
    x: NoPadding,
    y: NoPadding,
}

// CHECK-LABEL: fn cmp_recursive(
fn cmp_recursive(a: &Recursive, b: &Recursive) -> bool {
    // CHECK: compare_bitwise::<Recursive>(
    a == b
}
// EMIT_MIR derived_partialeq.cmp_recursive.PreCodegen.after.mir

#[derive(PartialEq)]
pub struct NoPaddingArray {
    x: [u16; 2],
    y: u16,
}

// CHECK-LABEL: fn cmp_no_padding_array(
fn cmp_no_padding_array(a: &NoPaddingArray, b: &NoPaddingArray) -> bool {
    // CHECK: compare_bitwise::<NoPaddingArray>(
    a == b
}
// EMIT_MIR derived_partialeq.cmp_no_padding_array.PreCodegen.after.mir

#[derive(PartialEq)]
pub struct NoPaddingAfterOpt {
    x: u8,
    y: u16,
    z: u8,
}

// CHECK-LABEL: fn cmp_no_padding_after_opt(
fn cmp_no_padding_after_opt(a: &NoPaddingAfterOpt, b: &NoPaddingAfterOpt) -> bool {
    // CHECK: compare_bitwise::<NoPaddingAfterOpt>(
    a == b
}
// EMIT_MIR derived_partialeq.cmp_no_padding_after_opt.PreCodegen.after.mir

#[derive(PartialEq)]
pub struct HasPadding {
    x: u8,
    y: u16,
}

// CHECK-LABEL: fn cmp_has_padding(
fn cmp_has_padding(a: &HasPadding, b: &HasPadding) -> bool {
    // CHECK-NOT: compare_bitwise
    a == b
}
// EMIT_MIR derived_partialeq.cmp_has_padding.PreCodegen.after.mir

#[derive(PartialEq)]
pub struct HasFloat {
    x: f32,
    y: u32,
}

// CHECK-LABEL: fn cmp_has_float(
fn cmp_has_float(a: &HasFloat, b: &HasFloat) -> bool {
    // CHECK-NOT: compare_bitwise
    a == b
}
// EMIT_MIR derived_partialeq.cmp_has_float.PreCodegen.after.mir
