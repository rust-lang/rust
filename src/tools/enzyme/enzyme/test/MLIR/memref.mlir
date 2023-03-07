// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %c0 = arith.constant 0 : index
    %tmp = memref.alloc() : memref<1xf64>
    %y = arith.mulf %x, %x : f64
    memref.store %y, %tmp[%c0] : memref<1xf64>
    %r = memref.load %tmp[%c0] : memref<1xf64>
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[i0:.+]] = memref.alloc() : memref<1xf64>
// CHECK-NEXT:     %[[i1:.+]] = memref.alloc() : memref<1xf64>
// CHECK-NEXT:     %[[i2:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i4:.+]] = arith.addf %[[i2]], %[[i3]] : f64
// CHECK-NEXT:     %[[i5:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:     memref.store %[[i4]], %[[i0]][%[[c0]]] : memref<1xf64>
// CHECK-NEXT:     memref.store %[[i5]], %[[i1]][%[[c0]]] : memref<1xf64>
// CHECK-NEXT:     %[[i6:.+]] = memref.load %[[i0]][%[[c0]]] : memref<1xf64>
// CHECK-NEXT:     %[[i7:.+]] = memref.load %[[i1]][%[[c0]]] : memref<1xf64>
// CHECK-NEXT:     return %[[i6]] : f64
// CHECK-NEXT:   }
