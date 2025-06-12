//@ test-mir-pass: CheckEnums

// EMIT_MIR enum_check_no_niche.main.CheckEnums.diff

enum Foo {
    A,
    B,
}

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK assert(copy .*, "trying to construct an enum from an invalid value {}", .*)
    let _val = unsafe { core::mem::transmute::<_, Foo>(0_u8) };
}
