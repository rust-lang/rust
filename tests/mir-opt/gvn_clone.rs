//@ test-mir-pass: GVN
//@ compile-flags: -Zmir-enable-passes=+InstSimplify-before-inline

// Check if we have transformed the default clone to copy in the specific pipeline.

// EMIT_MIR gvn_clone.{impl#0}-clone.GVN.diff

// CHECK-LABEL: ::clone(
// CHECK-NOT: = AllCopy { {{.*}} };
// CHECK: _0 = copy (*_1);
// CHECK: return;
#[derive(Clone)]
struct AllCopy {
    a: i32,
    b: u64,
    c: [i8; 3],
}
