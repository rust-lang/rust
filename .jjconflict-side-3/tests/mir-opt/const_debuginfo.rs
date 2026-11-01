//@ test-mir-pass: SingleUseConsts
//@ compile-flags: -C overflow-checks=no -Zmir-enable-passes=+GVN -Zdump-mir-exclude-alloc-bytes

#![allow(unused)]

struct Point {
    x: u32,
    y: u32,
}

// EMIT_MIR const_debuginfo.main.SingleUseConsts.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => const 1_u8;
    // CHECK: debug y => const 2_u8;
    // CHECK: debug z => const 3_u8;
    // CHECK: debug sum => const 6_u8;
    // CHECK: debug s => const "hello, world!";
    // CHECK: debug f => {{_.*}};
    // CHECK: debug o => const Option::<u16>::Some(99_u16);
    // CHECK: debug p => const Point
    // CHECK: debug a => const 64_u32;
    let x = 1u8;
    let y = 2u8;
    let z = 3u8;
    let sum = x + y + z;

    let s = "hello, world!";

    let f = (true, false, 123u32);

    let o = Some(99u16);

    let p = Point { x: 32, y: 32 };
    let a = p.x + p.y;
}
