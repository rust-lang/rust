// compile-flags: --crate-type=lib -O -Cdebuginfo=2 -Cno-prepopulate-passes -Zmir-enable-passes=-ScalarReplacementOfAggregates
// MIR SROA will decompose the closure
// min-llvm-version: 15.0 # this test uses opaque pointer notation
#![feature(stmt_expr_attributes)]

pub struct S([usize; 8]);

#[no_mangle]
pub fn outer_function(x: S, y: S) -> usize {
    (#[inline(always)]|| {
        let _z = x;
        y.0[0]
    })()
}

// Check that we do not attempt to load from the spilled arg before it is assigned to
// when generating debuginfo.
// CHECK-LABEL: @outer_function
// CHECK: [[spill:%.*]] = alloca %"[closure@{{.*.rs}}:10:23: 10:25]"
// CHECK-NOT: [[ptr_tmp:%.*]] = getelementptr inbounds %"[closure@{{.*.rs}}:10:23: 10:25]", ptr [[spill]]
// CHECK-NOT: [[load:%.*]] = load ptr, ptr
// CHECK: call void @llvm.lifetime.start{{.*}}({{.*}}, ptr [[spill]])
// CHECK: [[inner:%.*]] = getelementptr inbounds %"{{.*}}", ptr [[spill]]
// CHECK: call void @llvm.memcpy{{.*}}(ptr {{align .*}} [[inner]], ptr {{align .*}} %x
