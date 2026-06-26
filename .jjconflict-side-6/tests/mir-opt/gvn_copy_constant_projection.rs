// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

use std::cmp::Ordering;
fn compare_constant_index(x: [i32; 1], y: [i32; 1]) -> Ordering {
    // CHECK-LABEL: fn compare_constant_index(
    // CHECK-NOT: (*{{_.*}});
    // CHECK: [[lhs:_.*]] = copy _1[0 of 1];
    // CHECK-NOT: (*{{_.*}});
    // CHECK: [[rhs:_.*]] = copy _2[0 of 1];
    // CHECK: _0 = Cmp(move [[lhs]], move [[rhs]]);
    Ord::cmp(&x[0], &y[0])
}

fn main() {
    compare_constant_index([1], [2]);
}

// EMIT_MIR gvn_copy_constant_projection.compare_constant_index.GVN.diff
