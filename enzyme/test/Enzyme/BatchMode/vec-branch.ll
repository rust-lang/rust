; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define double @relu(double %x, double %a) {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %ax = fmul double %x, %a
  ret double %ax

cond.end:                                         ; preds = %entry, %cond.true
  ret double %x
}

define [4 x double] @vecrelu(double %x1, double %x2, double %x3, double %x4, double %a) {
entry:
  %0 = tail call [4 x double] (...) @__enzyme_batch(double (double, double)* nonnull @relu, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4, metadata !"enzyme_scalar", double %a)
  ret [4 x double] %0
}

declare [4 x double] @__enzyme_batch(...)


; XFAIL: -