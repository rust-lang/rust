//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes

// EMIT_MIR address_of_pair.fn0.GVN.diff
pub fn fn0() -> bool {
    // CHECK-LABEL: fn fn0(
    // CHECK: debug pair => [[pair:_.*]];
    // CHECK: debug ptr => [[ptr:_.*]];
    // CHECK: debug ret => [[ret:_.*]];
    // CHECK: (*[[ptr]]) = const true;
    // CHECK-NOT: = const false;
    // CHECK-NOT: = const true;
    // CHECK: [[tmp:_.*]] = copy ([[pair]].1: bool);
    // CHECK-NOT: = const false;
    // CHECK-NOT: = const true;
    // CHECK: [[ret]] = Not(move [[tmp]]);
    // CHECK-NOT: = const false;
    // CHECK-NOT: = const true;
    // CHECK: _0 = copy [[ret]];
    let mut pair = (1, false);
    let ptr = core::ptr::addr_of_mut!(pair.1);
    pair = (1, false);
    unsafe {
        *ptr = true;
    }
    let ret = !pair.1;
    return ret;
}

pub fn main() {
    println!("{}", fn0());
}
