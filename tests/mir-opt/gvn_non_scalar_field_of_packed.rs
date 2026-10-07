//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes -Zverbose-internals

// This is a regression test for https://github.com/rust-lang/rust/issues/163782

#![crate_type = "lib"]

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Packed {
    pub a: u8,
    pub b: (u32, u32),
}

#[derive(Clone, Copy)]
pub struct Outer {
    pub x: u32,
    pub p: Packed,
}

const C: Outer = Outer { x: 7, p: Packed { a: 1, b: (2, 3) } };

// EMIT_MIR gvn_non_scalar_field_of_packed.non_scalar_field_of_packed.GVN.diff
pub fn non_scalar_field_of_packed() -> (u32, u32) {
    // CHECK-LABEL: fn non_scalar_field_of_packed(
    // CHECK: _0 = const ConstValue(Indirect { alloc_id: {{.*}}, offset: Size(0 bytes) }: (u32, u32));
    C.p.b
}
