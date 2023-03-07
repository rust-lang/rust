// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %c1_i64 = arith.constant 1 : i64
    %tmp = llvm.alloca %c1_i64 x f64 : (i64) -> !llvm.ptr<f64>
    %y = arith.mulf %x, %x : f64
    llvm.store %y, %tmp : !llvm.ptr<f64>
    %r = llvm.load %tmp : !llvm.ptr<f64>
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-NEXT:     %[[c1_i64:.+]] = arith.constant 1 : i64
// CHECK-NEXT:     %[[i0:.+]] = llvm.alloca %[[c1_i64]] x f64 : (i64) -> !llvm.ptr<f64>
// CHECK-NEXT:     %[[i1:.+]] = llvm.alloca %[[c1_i64]] x f64 : (i64) -> !llvm.ptr<f64>
// CHECK-NEXT:     %[[i2:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i4:.+]] = arith.addf %[[i2]], %[[i3]] : f64
// CHECK-NEXT:     %[[i5:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:     llvm.store %[[i4]], %[[i0]] : !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %[[i5]], %[[i1]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[i6:.+]] = llvm.load %[[i0]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[i7:.+]] = llvm.load %[[i1]] : !llvm.ptr<f64>
// CHECK-NEXT:     return %[[i6]] : f64
// CHECK-NEXT:   }
