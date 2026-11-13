//@ revisions: NONE TINY ALL
//@[NONE] compile-flags: -Zmir_strip_debuginfo=none
//@[TINY] compile-flags: -Zmir_strip_debuginfo=locals-in-tiny-functions
//@[ALL] compile-flags: -Zmir_strip_debuginfo=all-locals

// CHECK: fn tiny_function
fn tiny_function(end: u32) -> u32 {
    // CHECK: debug end => _1;
    // NONE: debug a =>
    // NONE: debug b =>
    // TINY-NOT: debug a =>
    // TINY-NOT: debug b =>
    // ALL-NOT: debug a =>
    // ALL-NOT: debug b =>
    let a = !end;
    let b = a ^ 1;
    b
}

#[inline(never)]
fn opaque(_: u32) {}

// CHECK: fn looping_function
fn looping_function(end: u32) {
    // CHECK: debug end => _1;
    // NONE: debug i =>
    // NONE: debug x =>
    // TINY: debug i =>
    // TINY: debug x =>
    // ALL-NOT: debug i =>
    // ALL-NOT: debug x =>
    let mut i = 0;
    while i < end {
        let x = i ^ 1;
        opaque(x);
    }
}

fn main() {}
