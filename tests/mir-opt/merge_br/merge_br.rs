//@ test-mir-pass: MergeBranchSimplification
//@ compile-flags: -Cdebuginfo=2

pub enum NoFields {
    A,
    B,
}

pub enum NoFields2 {
    A,
    B,
}

// EMIT_MIR merge_br.no_fields.MergeBranchSimplification.diff
pub fn no_fields(a: NoFields) -> NoFields {
    // CHECK-LABEL: no_fields(
    // CHECK: bb0: {
    // CHECK-NEXT: _{{.*}} = discriminant([[SRC:_1]]);
    // CHECK-NEXT: _0 = copy [[SRC]];
    match a {
        NoFields::A => NoFields::A,
        NoFields::B => NoFields::B,
    }
}

// EMIT_MIR merge_br.no_fields_ref.MergeBranchSimplification.diff
pub fn no_fields_ref(a: &NoFields) -> NoFields {
    // CHECK-LABEL: no_fields_ref(
    // CHECK: bb0: {
    // CHECK-NEXT: _{{.*}} = discriminant([[SRC:\(\*_1\)]]);
    // CHECK-NEXT: _0 = copy [[SRC]];
    match a {
        NoFields::A => NoFields::A,
        NoFields::B => NoFields::B,
    }
}

// EMIT_MIR merge_br.no_fields_mismatch_type_failed.MergeBranchSimplification.diff
pub fn no_fields_mismatch_type_failed(a: NoFields) -> NoFields2 {
    // CHECK-LABEL: no_fields_mismatch_type_failed(
    // CHECK: bb0: {
    // CHECK-NEXT: _{{.*}} = discriminant([[SRC:_1]]);
    // CHECK-NOT: _0 = copy [[SRC]];
    match a {
        NoFields::A => NoFields2::A,
        NoFields::B => NoFields2::B,
    }
}

// EMIT_MIR merge_br.no_fields_failed.MergeBranchSimplification.diff
pub fn no_fields_failed(a: NoFields) -> NoFields {
    // CHECK-LABEL: no_fields_failed(
    // CHECK: bb0: {
    // CHECK-NEXT: _{{.*}} = discriminant([[SRC:_1]]);
    // CHECK-NOT: _0 = copy [[SRC]];
    match a {
        NoFields::A => NoFields::B,
        NoFields::B => NoFields::A,
    }
}

// EMIT_MIR merge_br.option.MergeBranchSimplification.diff
pub fn option(a: Option<i32>) -> Option<i32> {
    // CHECK-LABEL: option(
    // CHECK: bb0: {
    // CHECK-NEXT: _{{.*}} = discriminant([[SRC:_1]]);
    // CHECK-NEXT: _0 = copy [[SRC]];
    match a {
        Some(_b) => a,
        None => None,
    }
}

// EMIT_MIR merge_br.option_dse_failed.MergeBranchSimplification.diff
pub fn option_dse_failed(a: Option<i32>, b: &mut i32) -> Option<i32> {
    // CHECK-LABEL: option_dse_failed(
    // CHECK: bb0: {
    // CHECK-NEXT: [[DISCR:_.*]] = discriminant([[SRC:_1]]);
    // CHECK-NEXT: switchInt(move [[DISCR]])
    match a {
        Some(_) => {
            *b = 1;
            a
        }
        None => None,
    }
}
