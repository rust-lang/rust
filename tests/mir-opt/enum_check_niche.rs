//@ test-mir-pass: CheckEnums

// EMIT_MIR enum_check_niche.main.CheckEnums.diff

#[repr(u16)]
enum Mix {
    A,
    B(u16),
}

enum Nested {
    C(Mix),
    D,
    E,
}

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK assert(copy .*, "trying to construct an enum from an invalid value {}", .*)
    let _val = unsafe { core::mem::transmute::<_, Nested>(0_u32) };
}
