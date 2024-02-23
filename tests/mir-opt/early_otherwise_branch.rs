//@ unit-test: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+UninhabitedEnumBranching

// We can't optimize it because y may be an invalid value.
// EMIT_MIR early_otherwise_branch.opt1.EarlyOtherwiseBranch.diff
fn opt1(x: Option<u32>, y: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt1(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: Ne
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Some(a), Some(b)) => 0,
        _ => 1,
    }
}

// EMIT_MIR early_otherwise_branch.opt2.EarlyOtherwiseBranch.diff
fn opt2(x: Option<u32>, y: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt2(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: [[CMP_LOCAL]] = Ne([[LOCAL1]], move [[LOCAL2]]);
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Some(a), Some(b)) => 0,
        (None, None) => 2,
        _ => 1,
    }
}

// optimize despite different types
// EMIT_MIR early_otherwise_branch.opt3.EarlyOtherwiseBranch.diff
fn opt3(x: Option<u32>, y: Option<bool>) -> u32 {
    // CHECK-LABEL: fn opt3(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: [[CMP_LOCAL]] = Ne([[LOCAL1]], move [[LOCAL2]]);
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Some(a), Some(b)) => 0,
        (None, None) => 2,
        _ => 1,
    }
}

// EMIT_MIR early_otherwise_branch.opt4.EarlyOtherwiseBranch.diff
fn opt4(x: u32, y: u32) -> u32 {
    // CHECK-LABEL: fn opt4(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[CMP_LOCAL]] = Ne(
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (1, 1) => 4,
        (2, 2) => 5,
        (3, 3) => 6,
        _ => 0,
    }
}

// EMIT_MIR early_otherwise_branch.opt5.EarlyOtherwiseBranch.diff
fn opt5(x: u32, y: u32) -> u32 {
    // CHECK-LABEL: fn opt5(
    // CHECK: bb0: {
    // CHECK-NOT: Ne(
    // CHECK: switchInt(
    // CHECK-NEXT: }
    match (x, y) {
        (1, 1) => 4,
        (2, 2) => 5,
        (3, 2) => 6,
        _ => 0,
    }
}

// EMIT_MIR early_otherwise_branch.opt6.EarlyOtherwiseBranch.diff
fn opt6(x: u32, y: u32) -> u32 {
    // CHECK-LABEL: fn opt6(
    // CHECK: bb0: {
    // CHECK: switchInt((_{{.*}}: u32)) -> [10: [[SWITCH_BB:bb.*]], otherwise: [[OTHERWISE:bb.*]]];
    // CHECK-NEXT: }
    // CHECK: [[SWITCH_BB]]:
    // CHECK:  switchInt((_{{.*}}: u32)) -> [1: bb{{.*}}, 2: bb{{.*}}, 3: bb{{.*}}, otherwise: [[OTHERWISE]]];
    // CHECK-NEXT: }
    match (x, y) {
        (1, 10) => 4,
        (2, 10) => 5,
        (3, 10) => 6,
        _ => 0,
    }
}

// EMIT_MIR early_otherwise_branch.opt7.EarlyOtherwiseBranch.diff
fn opt7(x: Option<u32>, y: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt7(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: switchInt(move [[LOCAL2]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Some(a), Some(b)) => 0,
        (None, Some(b)) => 2,
        _ => 1,
    }
}

// EMIT_MIR early_otherwise_branch.opt8.EarlyOtherwiseBranch.diff
fn opt8(x: u32, y: u64) -> u32 {
    // CHECK-LABEL: fn opt8(
    // CHECK: bb0: {
    // CHECK: switchInt((_{{.*}}: u64)) -> [10: [[SWITCH_BB:bb.*]], otherwise: [[OTHERWISE:bb.*]]];
    // CHECK-NEXT: }
    // CHECK: [[SWITCH_BB]]:
    // CHECK:  switchInt((_{{.*}}: u32)) -> [1: bb{{.*}}, 2: bb{{.*}}, 3: bb{{.*}}, otherwise: [[OTHERWISE]]];
    // CHECK-NEXT: }
    match (x, y) {
        (1, 10) => 4,
        (2, 10) => 5,
        (3, 10) => 6,
        _ => 0,
    }
}

#[repr(u8)]
enum E8 {
    A,
    B,
    C,
}

#[repr(u16)]
enum E16 {
    A,
    B,
    C,
}

// Can we add a cast instruction for transformation?
// EMIT_MIR early_otherwise_branch.opt9.EarlyOtherwiseBranch.diff
fn opt9(x: E8, y: E16) -> u32 {
    // CHECK-LABEL: fn opt9(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (E8::A, E16::A) => 1,
        (E8::B, E16::B) => 2,
        (E8::C, E16::C) => 3,
        _ => 0,
    }
}

// Since the target values are the same, we can optimize.
// EMIT_MIR early_otherwise_branch.opt10.EarlyOtherwiseBranch.diff
fn opt10(x: E8, y: E16) -> u32 {
    // CHECK-LABEL: fn opt10(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: switchInt(move [[LOCAL2]]) -> [0: [[SWITCH_BB:bb.*]], otherwise: [[OTHERWISE:bb.*]]];
    // CHECK-NEXT: }
    // CHECK: [[SWITCH_BB]]:
    // CHECK:  switchInt([[LOCAL1]]) -> [0: bb{{.*}}, 1: bb{{.*}}, 2: bb{{.*}}, otherwise: [[OTHERWISE]]];
    // CHECK-NEXT: }
    match (x, y) {
        (E8::A, E16::A) => 1,
        (E8::B, E16::A) => 2,
        (E8::C, E16::A) => 3,
        _ => 0,
    }
}

fn main() {
    opt1(None, Some(0));
    opt2(None, Some(0));
    opt3(None, Some(false));
    opt4(0, 0);
    opt5(0, 0);
    opt6(0, 0);
    opt7(None, Some(0));
    opt8(0, 0);
    opt9(E8::A, E16::A);
    opt10(E8::A, E16::A);
}
