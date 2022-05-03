; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

%struct.Gradients = type { [2 x double] }

declare %struct.Gradients @__enzyme_autodiff(double (double)*, ...)

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define %struct.Gradients @dsquare(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @square, metadata !"enzyme_width", i64 2, double %x)
  ret %struct.Gradients %0
}


; CHECK: define internal { [2 x double] } @diffe2square(double %x, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"x'de"
; CHECK-NEXT:   %0 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %m0diffex = fmul fast double %0, %x
; CHECK-NEXT:   %1 = insertvalue [2 x double] undef, double %m0diffex, 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %m0diffex1 = fmul fast double %2, %x
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 0
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i5]], %m0diffex
; CHECK-NEXT:   store double %[[i6]], double* %[[i4]]
; CHECK-NEXT:   %[[i7:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 1
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[i8]], %m0diffex1
; CHECK-NEXT:   store double %[[i9]], double* %[[i7]]
; CHECK-NEXT:   %[[i10:.+]] = load double, double* %[[i4]]
; CHECK-NEXT:   %[[i11:.+]] = fadd fast double %[[i10]], %m0diffex
; CHECK-NEXT:   store double %[[i11]], double* %[[i4]]
; CHECK-NEXT:   %[[i12:.+]] = load double, double* %[[i7]]
; CHECK-NEXT:   %[[i13:.+]] = fadd fast double %[[i12]], %m0diffex1
; CHECK-NEXT:   store double %[[i13]], double* %[[i7]]
; CHECK-NEXT:   %[[i14:.+]] = load [2 x double], [2 x double]* %"x'de"
; CHECK-NEXT:   %[[i15:.+]] = insertvalue { [2 x double] } undef, [2 x double] %[[i14]], 0
; CHECK-NEXT:   ret { [2 x double] } %[[i15]]
; CHECK-NEXT: }
