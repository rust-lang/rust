//@ test-mir-pass: InstSimplify-after-simplifycfg

// EMIT_MIR bool_compare.eq_true.InstSimplify-after-simplifycfg.diff
fn eq_true(x: bool) -> u32 {
    // CHECK-LABEL: fn eq_true(
    // CHECK-NOT: Eq(
    if x == true { 0 } else { 1 }
}

// EMIT_MIR bool_compare.true_eq.InstSimplify-after-simplifycfg.diff
fn true_eq(x: bool) -> u32 {
    // CHECK-LABEL: fn true_eq(
    // CHECK-NOT: Eq(
    if true == x { 0 } else { 1 }
}

// EMIT_MIR bool_compare.ne_true.InstSimplify-after-simplifycfg.diff
fn ne_true(x: bool) -> u32 {
    // CHECK-LABEL: fn ne_true(
    // CHECK: Not(
    if x != true { 0 } else { 1 }
}

// EMIT_MIR bool_compare.true_ne.InstSimplify-after-simplifycfg.diff
fn true_ne(x: bool) -> u32 {
    // CHECK-LABEL: fn true_ne(
    // CHECK: Not(
    if true != x { 0 } else { 1 }
}

// EMIT_MIR bool_compare.eq_false.InstSimplify-after-simplifycfg.diff
fn eq_false(x: bool) -> u32 {
    // CHECK-LABEL: fn eq_false(
    // CHECK: Not(
    if x == false { 0 } else { 1 }
}

// EMIT_MIR bool_compare.false_eq.InstSimplify-after-simplifycfg.diff
fn false_eq(x: bool) -> u32 {
    // CHECK-LABEL: fn false_eq(
    // CHECK: Not(
    if false == x { 0 } else { 1 }
}

// EMIT_MIR bool_compare.ne_false.InstSimplify-after-simplifycfg.diff
fn ne_false(x: bool) -> u32 {
    // CHECK-LABEL: fn ne_false(
    // CHECK-NOT: Ne(
    if x != false { 0 } else { 1 }
}

// EMIT_MIR bool_compare.false_ne.InstSimplify-after-simplifycfg.diff
fn false_ne(x: bool) -> u32 {
    // CHECK-LABEL: fn false_ne(
    // CHECK-NOT: Ne(
    if false != x { 0 } else { 1 }
}

fn main() {
    eq_true(false);
    true_eq(false);
    ne_true(false);
    true_ne(false);
    eq_false(false);
    false_eq(false);
    ne_false(false);
    false_ne(false);
}
