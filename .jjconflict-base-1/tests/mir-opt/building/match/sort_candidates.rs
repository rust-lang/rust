// Check specific cases of sorting candidates in match lowering.

// EMIT_MIR sort_candidates.constant_eq.SimplifyCfg-initial.after.mir
fn constant_eq(s: &str, b: bool) -> u32 {
    // Check that we only test "a" once

    // CHECK-LABEL: fn constant_eq(
    // CHECK-NOT: const "a"
    // CHECK: {{_[0-9]+}} = const "a" as &[u8] (Transmute);
    // CHECK-NOT: const "a"
    match (s, b) {
        ("a", _) if true => 1,
        ("b", true) => 2,
        ("a", true) => 3,
        (_, true) => 4,
        _ => 5,
    }
}

// EMIT_MIR sort_candidates.disjoint_ranges.SimplifyCfg-initial.after.mir
fn disjoint_ranges(x: i32, b: bool) -> u32 {
    // When `(0..=10).contains(x) && !b`, we should jump to the last arm without testing the two
    // other candidates.

    // CHECK-LABEL: fn disjoint_ranges(
    // CHECK: debug b => _2;
    // CHECK: bb0: {
    // CHECK: switchInt(copy _2) -> [0: [[jump:bb.*]], otherwise: {{bb.*}}];
    // CHECK: [[jump]]: {
    // CHECK-NEXT: _0 = const 3_u32;
    // CHECK-NEXT: return;
    match x {
        0..10 if b => 0,
        10..=20 => 1,
        -1 => 2,
        _ => 3,
    }
}

fn main() {}
