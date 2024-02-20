#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

// EMIT_MIR never_patterns.opt1.SimplifyCfg-initial.after.mir
fn opt1(res: &Result<u32, Void>) -> &u32 {
    // CHECK-LABEL: fn opt1(
    // CHECK: bb0: {
    // CHECK-NOT: {{bb.*}}: {
    // CHECK: return;
    match res {
        Ok(x) => x,
        Err(!),
    }
}

// EMIT_MIR never_patterns.opt2.SimplifyCfg-initial.after.mir
fn opt2(res: &Result<u32, Void>) -> &u32 {
    // CHECK-LABEL: fn opt2(
    // CHECK: bb0: {
    // CHECK-NOT: {{bb.*}}: {
    // CHECK: return;
    match res {
        Ok(x) | Err(!) => x,
    }
}

// EMIT_MIR never_patterns.opt3.SimplifyCfg-initial.after.mir
fn opt3(res: &Result<u32, Void>) -> &u32 {
    // CHECK-LABEL: fn opt3(
    // CHECK: bb0: {
    // CHECK-NOT: {{bb.*}}: {
    // CHECK: return;
    match res {
        Err(!) | Ok(x) => x,
    }
}

fn main() {
    assert_eq!(opt1(&Ok(0)), &0);
    assert_eq!(opt2(&Ok(0)), &0);
    assert_eq!(opt3(&Ok(0)), &0);
}
