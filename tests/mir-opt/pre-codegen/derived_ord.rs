//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

#![crate_type = "lib"]

#[derive(PartialOrd, PartialEq)]
pub struct MultiField(char, i16);

// Because this isn't derived by the impl, it's not on the `{impl#0}-partial_cmp`,
// and thus we need to call it to see what the inlined generic one produces.
pub fn demo_le(a: &MultiField, b: &MultiField) -> bool {
    // CHECK-LABEL: fn demo_le
    // CHECK: inlined <MultiField as PartialOrd>::le
    // CHECK: inlined{{.+}}is_some_and
    // CHECK: inlined <MultiField as PartialOrd>::partial_cmp

    // CHECK: [[A0:_[0-9]+]] = copy ((*_1).0: char);
    // CHECK: [[B0:_[0-9]+]] = copy ((*_2).0: char);
    // CHECK: Cmp(move [[A0]], move [[B0]]);

    // CHECK: [[D0:_[0-9]+]] = discriminant({{.+}});
    // CHECK: switchInt(move [[D0]]) -> [0: bb{{[0-9]+}}, otherwise: bb{{[0-9]+}}];

    // CHECK: [[A1:_[0-9]+]] = copy ((*_1).1: i16);
    // CHECK: [[B1:_[0-9]+]] = copy ((*_2).1: i16);
    // CHECK: Cmp(move [[A1]], move [[B1]]);

    // CHECK: [[D1:_[0-9]+]] = discriminant({{.+}});
    // CHECK: switchInt(move [[D1]]) -> [0: bb{{[0-9]+}}, 1: bb{{[0-9]+}}, otherwise: bb{{[0-9]+}}];

    // CHECK: [[D2:_[0-9]+]] = discriminant({{.+}});
    // CHECK: _0 = Le(move [[D2]], const 0_i8);
    *a <= *b
}

// EMIT_MIR derived_ord.{impl#0}-partial_cmp.PreCodegen.after.mir
// EMIT_MIR derived_ord.demo_le.PreCodegen.after.mir
