// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %cst = arith.constant 10.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %r = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (f64) {
      %n = arith.addf %arg2, %x : f64
      scf.yield %n : f64
    }
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[cst_0:.+]] = arith.constant 1.000000e+01 : f64
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c10:.+]] = arith.constant 10 : index
// CHECK-NEXT:     %[[i0:.+]]:2 = scf.for %[[arg2:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg3:.+]] = %[[cst_0]], %[[arg4:.+]] = %[[cst]]) -> (f64, f64) {
// CHECK-NEXT:       %[[i1:.+]] = arith.addf %[[arg4]], %[[arg1]] : f64
// CHECK-NEXT:       %[[i2:.+]] = arith.addf %[[arg3]], %[[arg0]] : f64
// CHECK-NEXT:       scf.yield %[[i2]], %[[i1]] : f64, f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[i0]]#1 : f64
// CHECK-NEXT:   }
