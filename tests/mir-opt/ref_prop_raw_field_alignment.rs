//@ test-mir-pass: ReferencePropagation
// EMIT_MIR ref_prop_raw_field_alignment.raw_reborrow_read.ReferencePropagation.diff
//@ edition: 2024

#![crate_type = "lib"]

#[repr(C, align(8))]
struct Aligned8 {
    field: u32,
}

fn raw_field_read(input: &[u64; 2]) -> u32 {
    // CHECK-LABEL: fn raw_field_read(
    // CHECK: [[FIELD:_.*]] = &raw const ((*{{_.*}}).0: u32);
    // CHECK-NOT: _0 = copy ((*
    // CHECK: _0 = copy (*[[FIELD]]);

    unsafe {
        let ptr = input.as_ptr().cast::<u8>().add(4).cast::<Aligned8>();
        let field = &raw const (*ptr).field;
        *field
    }
}

// A raw-pointer reborrow without additional projections can be propagated.
fn raw_reborrow_read(ptr: *const u32) -> u32 {
    // CHECK-LABEL: fn raw_reborrow_read(
    // CHECK-NOT: = &raw const (*_1);
    // CHECK: _0 = copy (*_1);

    unsafe {
        let rebound = &raw const *ptr;
        *rebound
    }
}
