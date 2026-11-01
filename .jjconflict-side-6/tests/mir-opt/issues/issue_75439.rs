// EMIT_MIR issue_75439.foo.MatchBranchSimplification.diff
//@ ignore-endian-big

use std::mem::transmute;

pub fn foo(bytes: [u8; 16]) -> Option<[u8; 4]> {
    // CHECK-LABEL: fn foo(
    // CHECK: bb2: {
    // CHECK-NEXT: 0: [[bb:bb[0-9]+]],
    // CHECK-SAME: {{[0-9]+}}: [[bb]],

    // big endian `u32`s
    let dwords: [u32; 4] = unsafe { transmute(bytes) };
    const FF: u32 = 0x0000_ffff_u32.to_be();
    if let [0, 0, 0 | FF, ip] = dwords { Some(unsafe { transmute(ip) }) } else { None }
}

fn main() {
    let _ = foo([0; 16]);
}
