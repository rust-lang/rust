// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64, %y : f64) -> f64 {
    %c = arith.cmpf ult, %x, %y : f64
    cf.cond_br %c, ^blk2(%x : f64), ^blk2(%y : f64)

   ^blk2(%r : f64):
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64, %y : f64, %dy : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx, %y, %dy) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>] } : (f64, f64, f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64, %[[arg2:.+]]: f64, %[[arg3:.+]]: f64) -> f64 {
// CHECK-NEXT:     %[[i0:.+]] = call @fwddiffesquare(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]]) : (f64, f64, f64, f64) -> f64
// CHECK-NEXT:     return %[[i0]] : f64
// CHECK-NEXT:   }
// CHECK:   func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64, %[[arg2:.+]]: f64, %[[arg3]]: f64) -> f64 {
// CHECK-NEXT:     %[[i0:.+]] = arith.cmpf ult, %[[arg0]], %[[arg2]] : f64
// CHECK-NEXT:     cf.cond_br %[[i0]], ^bb1(%[[arg0]], %[[arg1]] : f64, f64), ^bb1(%[[arg2]], %[[arg3]] : f64, f64)
// CHECK-NEXT:   ^bb1(%[[i1:.+]]: f64, %[[i2:.+]]: f64):  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:     return %[[i2]] : f64
// CHECK-NEXT:   }
