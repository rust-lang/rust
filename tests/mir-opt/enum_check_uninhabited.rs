//@ test-mir-pass: CheckEnums

// EMIT_MIR enum_check_uninhabited.main.CheckEnums.diff

#![feature(never_type)]
#![allow(invalid_value)]

#[allow(dead_code)]
enum Wrap {
    A(!),
}
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK assert(copy .*, "trying to construct an enum from an invalid value {}", .*)
    let _val = unsafe { core::mem::transmute::<(), Wrap>(()) };
}
